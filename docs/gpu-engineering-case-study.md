# GPU Environment Engineering Case Study

## Objective and constraint

The experiment coordinates three gated language models in one hosted notebook session:

| Role | Model | Loading mode recorded by the notebook |
| --- | --- | --- |
| Prefix generator | `mistralai/Mistral-7B-Instruct-v0.3` | 4-bit NF4 with double quantization and bfloat16 compute |
| Target | `google/medgemma-4b-it` | bfloat16 |
| Judge | `google/gemma-2-2b-it` | bfloat16 |

The model labels total roughly 13 billion parameters. The saved notebook metadata identifies a Kaggle `nvidiaTeslaT4` accelerator, Docker image version `31236`, and Python `3.12.12`. The engineering problem was fitting and coordinating those models within one T4 session while resolving incompatibilities in the hosted environment.

## Compatibility decisions

| Constraint or failure | Intervention | Engineering reason |
| --- | --- | --- |
| The hosted PyTorch stack did not match the required CUDA path | Remove `torch`, `torchvision` and `torchaudio`, purge pip's cache, then reinstall from the CUDA 12.1 wheel index | Prevent an old cached wheel from silently surviving the environment rebuild |
| Parts of the stack were incompatible with NumPy 2.x | Constrain only `numpy<2.0` | Keep the compatibility boundary narrow instead of freezing every unrelated package |
| Three models had to coexist on one GPU | Load the 7B generator in 4-bit NF4 with double quantization; retain bfloat16 for the 4B target and 2B judge | Spend reduced precision where it frees the most memory while keeping the evaluation models in bfloat16 |
| Quantization might be unavailable on another machine | Guard the bitsandbytes path and fall back to bfloat16 | Fail transparently and preserve a usable, higher-memory execution path |
| TorchDynamo compilation was unstable in the target session | Disable TorchDynamo for the notebook workload | Remove a failing compilation path that was not required for the evaluation |
| Sampling reached a faulty probability-kernel path | Use greedy target decoding with `do_sample=False` and `num_beams=1` | Avoid the failing kernel and make target decoding deterministic within the recorded environment |

These decisions are implemented in the notebook and summarized in `requirements.txt`. They are compatibility controls, not claims that every future CUDA, driver or package combination will behave identically.

## Preserved execution evidence

Jupyter's `execution_count` fields record the order in which retained cells ran. The notebook's post-bootstrap record is:

| Counter | Notebook index | Preserved evidence |
| ---: | ---: | --- |
| `In [2]` | 7 | bitsandbytes detected; generator loaded in 4-bit NF4; `4-bit mode: True` |
| `In [3]` | 8 | generator token and cache configuration |
| `In [4]` | 9 | MedGemma target loaded in bfloat16 without bitsandbytes |
| `In [5]` | 10 | Gemma judge loaded |
| `In [6]` | 12 | inference helpers defined |
| `In [7]` | 14 | deterministic judge and strict-output parser defined |
| `In [14]` | 16 | wrapper proposal logic defined |
| `In [16]` | 18 | evolutionary `ProAttack` engine defined |
| `In [17]` | 19 | seven-goal evaluation executed |
| `In [18]` | 21 | elite candidates inspected |
| `In [19]` | 23 | summary: 7 goals met the configured threshold; average best score 8.00 |
| `In [20–22]` | 25, 27, 29 | result figures generated |
| `In [23]` | 31 | aggregate report: 104 candidates, 14.9 per goal, best-score range 7–10 |

The dependency-installation cells and several intermediate cells do not retain execution counters. They may have been cleared during the environment restart or later editing, so they are treated as configuration source rather than proof of execution. The record also has missing counters `1`, `8–13` and `15`; no sequence is invented for those gaps.

The generated [`notebook-execution-manifest.json`](notebook-execution-manifest.json) contains the complete retained counter list, notebook and cell-source SHA-256 digests, output types, bounded text previews and the environment metadata. Regenerate it without importing ML libraries or executing notebook code:

```bash
python tools/notebook_execution_manifest.py \
  adversarial_prompt_optimization.ipynb \
  --output docs/notebook-execution-manifest.json
```

## Reproduction protocol

This notebook depends on gated models and a compatible GPU environment. Do not start with **Run all** in a fresh hosted session.

1. Select a Tesla T4 GPU session and confirm access to all three Hugging Face repositories.
2. Run the acceleration and PyTorch bootstrap cell.
3. Run the `numpy<2.0` compatibility cell.
4. Restart the kernel if PyTorch was already imported by the session, then authenticate to Hugging Face.
5. Run the import and model-loading cells, confirming the generator reports `4-bit mode: True` before continuing.
6. Run the evaluation cells in notebook order.
7. Save the executed notebook and regenerate the execution manifest above.

For a new report, also capture `nvidia-smi`, `python --version`, `python -m pip freeze` and `python -m pip check`. Those artifacts are necessary to compare a future run with the saved Kaggle execution.

## Evidence limits

The saved notebook supports claims about its configured T4 environment, model-loading modes, retained execution order and stored evaluation outputs. It does not preserve peak VRAM, wall-clock duration, driver version or a complete package lock from the original run. It therefore does not support claims of byte-identical reproduction across different GPU architectures or future library versions.

The reported `7/7` threshold result measures the notebook's own LLM-judge procedure on seven prompts. It is an experiment result, not a clinical-safety certification or a general benchmark of MedGemma.
