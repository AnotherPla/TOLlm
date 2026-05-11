# TOLlm

TOLlm is a lightweight large language model inference engine. The project is
designed as a compact learning-oriented implementation of common LLM serving
ideas, including continuous batching, paged KV cache management, tensor
parallel.

## Features

- Continuous batching scheduler with separate prefill and decode execution
  paths.
- Paged KV cache block management with block tables, slot mapping, reference
  counting, and prefix cache hashing.
- Tensor parallel linear, embedding, and LM head layers based on
  `torch.distributed`.
- FlashAttention integration for variable-length prefill and KV-cache decode.
- CUDA Graph capture for small-batch decode replay when eager execution is not
  enforced.
- Temperature sampling and simple batched generation API.
- Minimal example and benchmark scripts for local experiments.

## Requirements

- Python `>=3.10,<3.13`
- CUDA-capable GPU
- PyTorch `>=2.4.0`
- Triton `>=3.0.0`
- Transformers `>=4.51.0`
- FlashAttention
- NCCL-compatible distributed backend for tensor parallel execution

This project is intended to run in a Linux CUDA environment. Windows users may
need WSL2 or a Linux machine for FlashAttention and NCCL support.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/AnotherPla/TOLlm.git
cd TOLlm
pip install -e .
```

FlashAttention installation depends on your CUDA and PyTorch versions. If
`pip install -e .` cannot install it automatically, install the matching
FlashAttention wheel manually first.

## Quick Start

Update the model path in `examples/minimal.py` to point to a local Qwen3 model
directory, then run:

```bash
python examples/minimal.py
```

Minimal usage:

```python
from tollm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-0.6B",
    tensor_parallel_size=1,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
)

outputs = llm.generate(
    ["Introduce yourself."],
    sampling_params,
)

print(outputs[0]["text"])
```

## Benchmark

A simple synthetic benchmark is provided in `benchmarks/minibench.py`.

```bash
python benchmarks/minibench.py
```

The script creates random token prompts and measures decode throughput. Update
the model path and request sizes in the script before running it on your own
hardware.

## Current Limitations

- Qwen3 is the main supported model family. LLaMA support is currently a
  placeholder.
- Greedy, top-k, top-p, repetition penalty, and stop strings are not implemented
  yet.
- The HTTP server module is only a stub.
- Benchmarking is minimal and should be expanded for fair comparisons with
  Hugging Face Transformers or vLLM.
- Test coverage is still limited.

## Roadmap

- Add more sampling strategies such as top-k, top-p, and repetition penalty.
- Expand benchmark scripts and report throughput/latency on real workloads.
- Add unit tests for scheduler, block manager, sampler, and tensor parallel
  layers.
- Improve server API and expose an OpenAI-compatible generation endpoint.
- Add support for more model architectures.
- Refine paged attention and prefix cache behavior.

## License

This project is licensed under the terms in `LICENSE`.