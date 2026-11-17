"""Saved-profile Parakeet entry point; no model import until arguments are validated."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from sa1_boundaries import validate_maximum
from sa1_contract import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio_file', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--vad-python', required=True, type=Path)
    p.add_argument('--evidence-dir', required=True, type=Path)
    p.add_argument('--chunk-len', type=float, default=30)
    p.add_argument('--speech-aware', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=6)
    a = p.parse_args()
    validate_maximum(a.chunk_len)
    if a.device not in ('auto', 'cpu') and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):
        p.error('Unsupported device')
    if not 1 <= a.threads <= 64:
        p.error('Threads must be between 1 and 64')
    if a.output.exists():
        raise FileExistsError(a.output)
    evidence = a.evidence_dir
    evidence.mkdir(exist_ok=False)
    save(evidence/'effective-settings.json', dict(speech_aware=a.speech_aware,
         max_decoded_seconds=a.chunk_len, device=a.device, threads=a.threads,
         compute_type='float32', decoder='greedy_batch', graphs=False, batch_size=1,
         worker_sha256=sha(__file__), input_sha256=sha(a.audio_file), model_sha256=sha(a.model)))
    # Evidence stays beside the job's normal transcript, even if the temporary result is removed.
    if not a.speech_aware:
        from fixed_worker import transcribe_buffered
        transcribe_buffered(str(a.audio_file), str(a.output), a.chunk_len,
                            model_path=str(a.model), temp_dir=a.output.parent/'chunks',
                            device=a.device, threads=a.threads)
        save(evidence/'result.json', read(a.output))
        return
    stage = 'VAD'
    try:
        # VAD chooses cuts on a continuous canonical timeline. Convert supported
        # noncanonical inputs without trimming, silence removal or time compaction.
        import wave
        canonical = False
        try:
            with wave.open(str(a.audio_file), 'rb') as wav:
                canonical = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        except (wave.Error, EOFError):
            pass
        if not canonical:
            stage = 'audio_conversion'
            converted = a.output.parent/'parakeet-16k.wav'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(a.audio_file),
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-n', str(converted)], check=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(evidence/'conversion.json', dict(original_audio=str(a.audio_file), converted_sha256=sha(converted),
                 recipe='ffmpeg mono16k PCM16; entire timeline, no trimming or silence removal'))
            a.audio_file = converted
        stage = 'VAD'
        clock = time.perf_counter()
        subprocess.run([str(a.vad_python), str(Path(__file__).with_name('sa1_vad.py')),
                        '--input', str(a.audio_file.resolve()), '--output', str(evidence/'vad-map.json'),
                        '--maximum', str(a.chunk_len)], check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        vad = read(evidence/'vad-map.json')
        components = dict(vad_subprocess_and_map_validation_seconds=time.perf_counter()-clock,
                          vad_inference_nested_seconds=vad['vad_inference_seconds'])
        stage = 'pipeline'
        import librosa
        import soundfile as sf
        from sa1_pipeline import NativeEngine, run_pipeline
        run_pipeline(a.audio_file, a.output, evidence, vad['chunks'], vad['total_samples'],
                     load_audio=lambda path: librosa.load(str(path), sr=None, mono=True),
                     load_model=lambda: NativeEngine(a.model, evidence, a.device, a.threads),
                     write_chunk=lambda path, audio, sr: sf.write(str(path), audio, sr),
                     temp_dir=a.output.parent/'chunks', component_times=components, chunk_len=a.chunk_len)
    except BaseException as error:
        if not (evidence/'failure.json').exists():
            save(evidence/'failure.json', dict(stage=stage, error_type=type(error).__name__, error=str(error)))
        raise

if __name__ == '__main__':
    main()
