# Saved-profile speech-aware Parakeet chunking

For NVIDIA Parakeet profiles, the transcription settings dialog exposes:

| Field | Default | Meaning |
|---|---|---|
| Use speech-aware chunking (VAD) | Off | SA1 cut selection, context ownership and pinned conservative D1 postpass |
| Maximum chunk length (seconds) | 30 | Maximum decoded input, including context; finite5–120seconds |
| Device | Existing profile value | CPU, CUDA or automatic selection; float32 |
| CPU threads | 0 | Zero resolves to6; otherwise1–64 |

Old profiles remain fixed30 when the new fields are absent. No automatic enablement. The two JSON fields are `parakeet_speech_aware` and nullable `parakeet_max_chunk_seconds`. An explicit zero is invalid. Other model families ignore these fields. The maximum does not change WhisperX's VAD/chunk settings.

SA1 covers the complete input timeline. Silero CPU VAD chooses safe cuts; silence is not removed. At30seconds the core budget is29.4seconds, search starts20seconds after the previous cut, safe context is300ms and forced-cut context250ms. For other maxima the core budget subtracts600ms from the decoded maximum; smaller budgets use a proportionate search start. All decoded chunks are checked against the maximum, including context and the final endpoint.

The original timestamp matcher/core ownership and D1 rules are retained. D1 only removes an exact unique contiguous2–8-word phrase at a forced edge when the original timing/overlap/ambiguity gates pass. It never rescues or rewrites words. Legitimate edge repetitions remain a risk. The independent reader/meeting validation had no D1 removals; it does not prove that branch's benefit.

The worker retains raw chunks, same-raw build2 control, merge/phrase decisions, effective settings, model/input hashes, original offsets and component timings in `parakeet-evidence` beside the job transcript. Invalid timestamps fail with raw evidence preserved. Global character offsets and display segments are rebuilt from retained native words. The existing Execution Info dialog shows captured profile parameters, including the generic stored batch field; effective batch1/graphs-off/decoder settings are recorded by the worker, and must not be inferred from that stored batch value.

The adapter embeds `parakeet_worker/*.py` in the Go binary. Normal environment preparation installs/refreshes them in `profile-worker-v1`, even when the Python environment is cached. It does not replace existing `transcribe.py` or `transcribe_buffered.py`. Every job receives explicit arguments from its saved/captured parameters; no process-wide chunk threshold or benchmark activation variable enables this feature.

The supported tested environment is NeMo2.5.3/Parakeet TDT0.6B v3, Torch2.8.0+cu128, float32, native greedy_batch, decoder CUDA graphs off, effective batch1. The CPU VAD subprocess uses the existing sibling WhisperX environment with FasterWhisper1.2.1/Silero v6. Its source/model hashes are validated. Use the retained package/model manifests when provisioning; dependency mismatch fails explicitly rather than downloading a different detector. Larger decoded maxima have logic coverage but no4GB memory guarantee;30seconds is the verified T1200 setting. Noncanonical input is converted to whole-file mono16k PCM16 through FFmpeg, without trimming or time compaction.

Build the frontend with its lockfile and `npm ci`, then `npm run build`; copy `web/frontend/dist` into `internal/web/dist` before `go build ./cmd/server`. Go toolchain1.24.4 matches the upstream module. The owning workspace supplies `scripts/build_durable_scriberr.ps1` for this Windows procedure and an opt-in loopback instance; its daily runtime remains separate.

Portable pure logic tests:

```text
python tests/python/test_parakeet_worker.py
go test ./internal/models ./internal/transcription/...
go test ./tests -run TestAPIHandlerTestSuite
```

The workspace's saved-output integration suite adds enum serialization, frozen-map replay, export/offset and failure checks without ASR. Real validation is bounded to its seven separately commissioned jobs; do not automatically rerun a completed benchmark.
