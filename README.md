# Adversarial Testing of MedGemma

An evaluation harness that stress-tests **MedGemma-4B-IT** against automatically generated adversarial prompt prefixes, using one language model to propose the prefixes and a third to score the resulting responses.

Status: research prototype.

## Pipeline

Three models run concurrently inside a single hosted GPU session (Kaggle, Tesla T4):

| Role | Model | Precision |
| --- | --- | --- |
| Generator | `mistralai/Mistral-7B-Instruct-v0.3` | 4-bit NF4 |
| Target | `google/medgemma-4b-it` | bfloat16 |
| Judge | `google/gemma-2-2b-it` | bfloat16 |

The generator proposes a short prefix, the prefix is prepended to a user request, the target answers, and the judge scores that answer as strict JSON on a 0–10 scale. An evolutionary hill-climbing loop keeps the top-K prefixes as elites, mutates them into the next generation of candidates, and stops early once a candidate reaches the score threshold.

## Environment engineering

Holding roughly 13B parameters across three models in one notebook session is not possible with the runtime's default packages. Rather than accept the preinstalled environment, I rebuilt it in stages and pinned only what actually needed pinning.

**Staged dependency installation.** The acceleration layer (`bitsandbytes`, `accelerate`) is installed first. The runtime's preinstalled PyTorch is then removed outright and `torch`, `torchvision` and `torchaudio` are reinstalled from the CUDA 12.1 wheel index to match the available GPU driver. The pip cache is purged between the two steps so the resolver cannot silently reuse the incompatible wheels it had already downloaded.

**Selective version pinning.** The central compatibility break was NumPy 2.x, which parts of the ML stack could not yet support. Instead of rolling the whole environment back to an older snapshot, I isolated the incompatibility boundary and constrained only the offending package with `numpy<2.0`, leaving every other dependency at its current version. Pinning narrowly keeps the environment reproducible without giving up unrelated upgrades.

**Runtime fault diagnosis.** Two further faults surfaced during execution and were traced to their source rather than worked around blindly. TorchDynamo's compilation path proved unstable on the target hardware and is disabled via `TORCHDYNAMO_DISABLE=1` before any import. Sampling-based decoding hit a faulty probability kernel, so the target is queried with greedy decoding (`do_sample=False`, `num_beams=1`) — which has the additional benefit of making the pipeline's outputs deterministic and reproducible.

**Memory-aware model loading.** Precision is chosen per model against the GPU budget. The 7B generator is loaded in 4-bit NF4 with double quantization and a bfloat16 compute dtype, which is what makes room for the other two; the 4B target and 2B judge run in bfloat16. All three use `device_map="auto"` for automatic device placement. The quantization path sits behind a capability check that falls back cleanly to bfloat16 if `bitsandbytes` is unavailable, so the notebook still runs on hardware without quantization support.

## Setup

MedGemma and Gemma are gated repositories — request access on Hugging Face and authenticate before loading.

Install the PyTorch stack from the CUDA 12.1 index rather than PyPI:

```bash
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Restart the kernel after the PyTorch reinstall so the new binaries are picked up, then authenticate:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Run `adversarial_prompt_optimization.ipynb` top to bottom. The environment cells are marked and only need to run once per session.

## Repository structure

```
adversarial_prompt_optimization.ipynb   Full pipeline: setup, model loading, optimization loop, analysis
requirements.txt                        Dependencies, with the NumPy constraint documented
```

## Technologies

Python · PyTorch · Hugging Face Transformers · bitsandbytes · Accelerate · CUDA · NumPy · pandas · Matplotlib · Seaborn

## Skills demonstrated

- Dependency conflict resolution and selective version pinning
- CUDA-compatible environment configuration and reproducible ML environments
- GPU runtime fault diagnosis (compilation backend, decoding kernels)
- 4-bit NF4 quantization and GPU memory optimization
- Multi-model LLM orchestration and Hugging Face deployment of gated models
- Graceful degradation across heterogeneous hardware
- LLM-as-judge evaluation design with robust output parsing
- Evolutionary/hill-climbing black-box optimization
