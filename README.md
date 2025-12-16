# CDLM: Corrective Diffusion Language Models

A research framework for studying and improving **self-revision in diffusion language models**, with a focus on **error localization, confidence-based remasking, and corrective training**.

The codebase supports controlled code corruption and iterative refinement, enabling systematic evaluation of why standard diffusion LMs fail to correct minimal code errors (e.g., one-token bugs) on benchmarks such as HumanEval.

## Features

- Iterative code generation and **self-revision** with diffusion language models
- Controlled semantic corruption (single-token and structured bugs)
- Analysis of **confidence–correctness misalignment** in remasking
- Corrective diffusion training with supervision on corrupted-but-visible tokens
- Evaluation on HumanEval and revision-focused benchmarks


## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd CDLM

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the example scripts in the `examples/` directory:

```bash
# For LLaDA model
bash examples/test_human-eval_llada.sh

# For Dream model
bash examples/test_human-eval_dream.sh

# For Open-dLLM model
bash examples/test_human-eval_open-dllm.sh
```

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

