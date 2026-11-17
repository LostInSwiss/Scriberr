"""CPU Silero boundary selection, independent of recognition and references."""
from pathlib import Path
import argparse
import time
import wave
from sa1_boundaries import boundaries, validate_maximum
from sa1_contract import save, sha

def prepare(audio, maximum=30):
    validate_maximum(maximum)
    import numpy as np
    import faster_whisper.vad as vad
    model = Path(vad.__file__).parent/'assets/silero_vad_v6.onnx'
    if sha(model) != '4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2' or sha(vad.__file__) != '37a9c774aefdd3162d936b896c8dcf5571b2ed938d65bffecfd631770049a18d':
        raise ValueError('Speech-aware chunking requires the supported Silero v6 source/assets; see setup documentation')
    with wave.open(str(audio), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Speech-aware chunking requires converted mono16k PCM16')
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)/np.float32(32768)
    if not len(samples):
        raise ValueError('Empty audio timeline')
    clock = time.perf_counter()
    regions = vad.get_speech_timestamps(samples, vad.VadOptions(threshold=0.5, neg_threshold=0.35,
        min_speech_duration_ms=0, max_speech_duration_s=float('inf'), min_silence_duration_ms=200, speech_pad_ms=0), sampling_rate=16000)
    seconds = time.perf_counter()-clock
    regions = [dict(start=int(r['start']), end=int(r['end'])) for r in regions]
    session = vad.get_vad_model().session
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('VAD must use CPUExecutionProvider')
    return dict(regions=regions, chunks=boundaries(len(samples), regions, maximum), total_samples=len(samples),
                input_sha256=sha(audio), vad_inference_seconds=seconds,
                vad_source_sha256=sha(vad.__file__), vad_model_sha256=sha(model),
                provider=session.get_providers(), inter_op_threads=session.get_session_options().inter_op_num_threads,
                intra_op_threads=session.get_session_options().intra_op_num_threads)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--maximum', type=float, default=30)
    a = p.parse_args()
    save(a.output, prepare(a.input, a.maximum))
