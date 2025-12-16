# CDLM: Code Diffusion Language Model

A code generation and refinement framework for evaluating and improving code generation models on benchmarks like HumanEval.

## Features

- Code generation with multiple model support (LLaDA, Dream, Open-dLLM)
- Code refinement using self-confidence remasking
- Evaluation on HumanEval benchmark
- Noise injection for robustness testing

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- Required Python packages (see requirements below)

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

