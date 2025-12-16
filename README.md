# CDLM: Corrective Diffusion Language Models

A research framework for studying and improving **self-revision in masked diffusion language models**, with a focus on **error localization, confidence-based refinement, and corrective training**.

The codebase supports controlled code corruption and iterative refinement, enabling systematic evaluation of how well masked diffusion language models can localize and correct errors through self-revision mechanisms.

## Features

- Iterative code generation and **self-revision** with diffusion language models
- Controlled token-level corruption
- Analysis of **confidence–correctness misalignment** in remasking
- Corrective training with supervision on corrupted-but-visible tokens
- Supports revision datasets: HumanEval, HumanEval+, MBPP, MBPP+ 


## Quick Start

### Installation

```bash
# Clone the repository
cd CDLM

# Install dependencies
pip install -r requirements.txt
```

### Usage

The example scripts run a complete pipeline: code generation → corruption with controlled errors → iterative refinement → evaluation.

```bash
# For LLaDA model
bash examples/test_human-eval_llada.sh

# For Dream model
bash examples/test_human-eval_dream.sh

# For Open-dLLM model
bash examples/test_human-eval_open-dllm.sh
```

### Key Parameters

You can modify the following parameters in the example scripts:

- **`MODEL_NAME`**: HuggingFace model identifier (e.g., `GSAI-ML/LLaDA-8B-Base`)
- **`DATASET`**: Evaluation dataset (`human-eval`, `human-eval+`, `mbpp+`)
- **`ERROR_TYPE`**: Type of corruption to inject
  - `operator`: Arithmetic/logical operator substitution
  - `var`: Identifier substitution (variable/function names)
  - `literal`: Constant value substitution
- **`N_REPLACE`**: Number of tokens to corrupt per sample (default: `1`)
- **`DATA_NUM`**: Number of corrupted variants to generate per correct sample (default: `5`)
- **`REFINED_STEPS`**: Number of denoising steps for refinement (starts from `2`; can use `2`, `3`, `4`, `5`, etc.)
- **`TEMPERATURE`**: Sampling temperature for refinement (default: `0.0` for greedy decoding)
- **`ALGORITHM`**: Refinement algorithm (`self_conf-remask:vanilla` for confidence-based remasking)
- **`CONFIDENCE_THRESHOLD`**: Confidence threshold for remasking decisions (default: `0.90`)

## Project Structure

```
CDLM/
├── codecorrection/      # Code correction and noise injection utilities
├── examples/            # Example scripts for running experiments
├── evaluate_code.py     # Code evaluation script
├── llada_sample.py      # Core sampling and remasking logic
├── refine_code.py       # Code refinement pipeline
├── sanitize.py          # Code sanitization utilities
└── utils.py             # Utility functions
```

## License

[Add your license here]

## Citation

[Add citation information here]

