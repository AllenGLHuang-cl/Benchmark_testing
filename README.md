# Benchmark Testing

Batch-driving scripts that generate images against either a local [ComfyUI](https://github.com/comfy-org/ComfyUI)
server or hosted model APIs (currently Gemini), and (in benchmark mode) measure per-request time
(and peak GPU memory, for the ComfyUI path) — then auto-build a PowerPoint comparing models
against each other.

![Automated Model Benchmark Pipeline](asset/auto_benchmark_eval.png)

Only the left side (preparing a dataset, a ComfyUI workflow, and weights for a new model) is manual.
Everything right of the dashed line — generation, benchmarking, and the comparison deck — is driven
by the scripts in this repo and the `model-benchmark-ppt` skill, and repeats automatically for every
new benchmark run.

## Quick start

```bash
python websocket_edit.py              # i2i batch over a features-layout dataset (ComfyUI)
python websocket_tti.py               # t2i batch over one flat prompt list, N models (ComfyUI)
python websocket_tti_structured.py    # t2i batch for structured-JSON-prompt models (ComfyUI, ideogram4)

export GEMINI_API_KEY=...             # never hardcode — see gemini_client.py
export GEMINI_BASE_URL=...
python gemini_edit.py                 # i2i batch over a features-layout dataset (Gemini API)
python gemini_tti.py                  # t2i batch over one flat prompt list (Gemini API)
```

Each script is configured by editing the constants in its `__main__` block (server address,
dataset path, model list, seed, etc.) — there's no CLI. See
[`CLAUDE.md`](CLAUDE.md) for the full architecture, the benchmark-recording conventions, and
known gotchas before making changes.
