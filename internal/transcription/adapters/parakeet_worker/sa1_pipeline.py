"""Isolated SA1 Scriberr worker. No model import/load until explicit execution gate."""
from pathlib import Path
import argparse
import datetime
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from sa1_contract import json_dumps, read, save, sha
from sa1_merge import merge_chunks
from sa1_duplicates import revise, export

def event(folder, name, **fields):
    encoded = json_dumps(dict(event=name, at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields))
    with (Path(folder)/'worker-events.jsonl').open('a', encoding='utf-8') as f:
        f.write(encoded+'\n')

def run_pipeline(audio_path, output, evidence, chunks, total, *, load_audio, load_model,
                 write_chunk, temp_dir, component_times=None, chunk_len=30):
    """Injectable orchestration. Synthetic tests supply stubs; no default ASR callable."""
    output, evidence, temp_dir = Path(output), Path(evidence), Path(temp_dir)
    times = dict(component_times or {})
    raws, recognition_times = [], []
    stage = 'audio_decode'
    try:
        clock = time.perf_counter()
        audio, sr = load_audio(audio_path)
        if sr != 16000 or len(audio) != total:
            raise ValueError('NeMo-side audio decode does not match frozen timeline')
        times['audio_decode_seconds'] = time.perf_counter()-clock
        stage = 'model_restore'
        clock = time.perf_counter()
        engine = load_model()
        times['model_initialization_seconds'] = time.perf_counter()-clock
        times['model_restore_nested_seconds'] = engine.restore_seconds
        event(evidence, 'model_ready', **engine.observe())
        temp_dir.mkdir(exist_ok=True)
        io_seconds = 0.0
        for c in chunks:
            stage = 'chunk_'+str(c['chunk'])
            path = temp_dir/f'chunk_{os.getpid()}_{c["chunk"]}.wav'
            if path.exists():
                raise FileExistsError(path)
            try:
                clock = time.perf_counter()
                write_chunk(path, audio[c['decode_start_sample']:c['decode_end_sample']], sr)
                io_seconds += time.perf_counter()-clock
                event(evidence, 'recognition_start', chunk=c['chunk'], bounds=c, effective=engine.observe())
                clock = time.perf_counter()
                raw = engine.transcribe_chunk(path)
                elapsed = time.perf_counter()-clock
                recognition_times.append(dict(chunk=c['chunk'], seconds=elapsed))
                # Save untouched native results before attempting any reconciliation.
                save(evidence/f'chunk-{c["chunk"]:02d}-native.json', raw, raw=True)
                raws.append(raw)
                event(evidence, 'recognition_end', chunk=c['chunk'], seconds=elapsed, effective=engine.observe())
            finally:
                if path.exists():
                    clock = time.perf_counter()
                    path.unlink()
                    io_seconds += time.perf_counter()-clock
        stage = 'reconciliation'
        clock = time.perf_counter()
        merged, audit = merge_chunks(chunks, raws, total, chunk_len)
        times['build2_control_reconciliation_seconds'] = time.perf_counter()-clock
        save(evidence/'control-build2.json', merged)
        save(evidence/'build2-merge-audit.json', audit)
        d1_clock = time.perf_counter()
        kept, phrases, boundaries = revise(chunks, audit)
        merged = export(kept, chunks)
        times['D1_postpass_seconds'] = time.perf_counter()-d1_clock
        save(evidence/'d1-decisions.json', dict(phrases=phrases, boundaries=boundaries))
        audit = dict(audit, accepted_words=kept, D1_phrases=phrases, D1_boundaries=boundaries)
        times['reconciliation_seconds'] = time.perf_counter()-clock
        save(evidence/'merge-audit.json', audit)
        times['chunk_io_seconds'] = io_seconds
        times['recognition_calls'] = recognition_times
        times['recognition_seconds'] = sum(r['seconds'] for r in recognition_times)
        stage = 'export'
        clock = time.perf_counter()
        result = dict(**merged, language='en', audio_file=str(audio_path), model='parakeet-tdt-0.6b-v3',
                      buffered=True, chunk_duration_secs=chunk_len, num_chunks=len(chunks), sa1_candidate='SA1-D1')
        save(output, result)
        save(evidence/'result.json', result)
        times['export_seconds'] = time.perf_counter()-clock
        save(evidence/'components.json', times)
        return result
    except BaseException as error:
        save(evidence/'failure.json', dict(stage=stage, status='failed_or_unmergeable', error_type=type(error).__name__,
             error=str(error), completed_chunk_calls=len(raws), completed_call_times=recognition_times, components=times))
        raise

class NativeEngine:
    def __init__(self, model_path, evidence, device="auto", threads=6):
        # This constructor is never called by source/synthetic checks.
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf, open_dict
        self.torch, self.OmegaConf = torch, OmegaConf
        torch.set_num_threads(threads)
        self.expected_device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        clock = time.perf_counter()
        self.model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=self.expected_device)
        self.restore_seconds = time.perf_counter()-clock
        config = self.model.cfg.decoding
        if config.strategy != 'greedy_batch':
            raise ValueError('Model default decoder differs from baseline')
        with open_dict(config.greedy):
            config.greedy['use_cuda_graph_decoder'] = False
        self.model.change_decoding_strategy(config)
        self.hooks = []
        observed = set()
        def hook(label):
            def observe(module, args, result):
                def tensors(x):
                    if isinstance(x, torch.Tensor):
                        return [str(x.dtype)+'@'+str(x.device)]
                    if isinstance(x, (list, tuple)):
                        return [t for y in x for t in tensors(y)]
                    if isinstance(x, dict):
                        return [t for y in x.values() for t in tensors(y)]
                    return []
                dtypes = tuple(tensors(result))
                key = (label, dtypes, torch.is_autocast_enabled('cuda'))
                if key not in observed:
                    observed.add(key)
                    event(evidence, 'operation_precision', module=label, output_dtypes=dtypes, cuda_autocast=key[2])
                # Forward hooks must return None: outputs are unchanged.
            return observe
        self.hooks.append(self.model.encoder.register_forward_hook(hook('encoder')))
        for cls in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.LSTM):
            for name, module in self.model.named_modules():
                if isinstance(module, cls):
                    self.hooks.append(module.register_forward_hook(hook(name)))
                    break
        self.observe()

    def observe(self):
        torch, model = self.torch, self.model
        parameter = next(model.parameters())
        dec = model.decoding.decoding
        computer = getattr(dec, 'decoding_computer', None)
        cfg = self.OmegaConf.to_container(model.cfg.decoding, resolve=True)
        if str(parameter.device) != getattr(self, 'expected_device', 'cuda:0') or str(parameter.dtype) != 'torch.float32':
            raise ValueError('SA1 requires baseline CUDA0 float32; no device/precision fallback')
        if torch.is_autocast_enabled('cuda'):
            raise ValueError('Baseline thread/autocast policy mismatch')
        if type(dec).__name__ != 'GreedyBatchedTDTInfer' or cfg['strategy'] != 'greedy_batch' or cfg['greedy']['use_cuda_graph_decoder'] is not False:
            raise ValueError('Baseline decoder configuration mismatch')
        if getattr(computer, 'cuda_graphs_mode', None) is not None:
            raise ValueError('Decoder graphs unexpectedly effective')
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Stored floating parameter dtype changed')
        return dict(device=str(parameter.device), parameter_dtype=str(parameter.dtype), threads=torch.get_num_threads(),
                    decoder_class=type(dec).__name__, graph_mode=str(getattr(computer, 'cuda_graphs_mode', None)),
                    decoding_config=cfg, batch_size=1, cuda_autocast=False)

    def transcribe_chunk(self, path):
        output = self.model.transcribe([str(path)], batch_size=1, timestamps=True)
        result = output[0]
        timestamps = getattr(result, 'timestamp', None)
        return dict(text=result.text, words=timestamps.get('word') if timestamps else None,
                    segments=timestamps.get('segment') if timestamps else None)

