# CDLM: Corrective Diffusion Language Models

<div align="center">

# CDLM: Corrective Diffusion Language Models

**A research framework for studying and improving self-revision in masked diffusion language models**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/zhangshuibai/CDLM)

</div>

<p align="center">
  <img src="figures/CRB_pipeline.pdf" alt="CRB Pipeline" width="800"/>
  <br>
  <em>Figure: Code Revision Benchmark (CRB) pipeline. <a href="figures/CRB_pipeline.pdf">View full PDF</a></em>
</p>

---

## Abstract

Diffusion language models are structurally well-suited for iterative error correction, as their non-causal denoising dynamics allow arbitrary positions in a sequence to be revised. However, standard masked diffusion language model (MDLM) training fails to induce such behavior, since incorrect but visible tokens receive no supervision, leaving token-level confidence uninformative about reliability. As a result, confidence-guided refinement is ineffective for targeted correction.

To address this mismatch, we study corrective behavior in diffusion language models and propose a **correction-oriented training principle** that explicitly supervises visible incorrect tokens, enabling MDLMs to acquire error-aware confidence and targeted refinement capabilities. To systematically evaluate corrective behavior, we introduce the **Code Revision Benchmark (CRB)**, a controllable and executable benchmark designed to evaluate corrective behavior in diffusion language models.

This repository provides the implementation and evaluation framework for our corrective diffusion language models, supporting controlled code corruption and iterative refinement for systematic evaluation of how well masked diffusion language models can localize and correct errors through self-revision mechanisms.

## Features

- 🔄 **Iterative self-revision** with diffusion language models
- 🎯 **Controlled token-level corruption** (operator, identifier, literal substitutions)
- 📊 **Confidence–correctness analysis** in remasking decisions
- 🎓 **Corrective training** with supervision on corrupted-but-visible tokens
- 📈 **Comprehensive evaluation** on HumanEval, HumanEval+, MBPP, and MBPP+

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zhangshuibai/CDLM.git
cd CDLM

# Install dependencies
pip install -r requirements.txt
```

### Usage

The example scripts run a complete pipeline: **code generation → corruption with controlled errors → iterative refinement → evaluation**.

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
│   └── generate.py      # Controlled corruption generation
├── examples/            # Example scripts for running experiments
│   ├── test_human-eval_llada.sh
│   ├── test_human-eval_dream.sh
│   └── test_human-eval_open-dllm.sh
├── figures/             # Paper figures and diagrams
│   └── CRB_pipeline.pdf
├── evaluate_code.py     # Code evaluation script
├── llada_sample.py      # Core sampling and remasking logic
├── refine_code.py       # Code refinement pipeline
├── sanitize.py          # Code sanitization utilities
└── utils.py             # Utility functions
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2024corrective,
  title={Corrective Diffusion Language Models},
  author={Zhang, Shuibai and Peng, Fred Zhangzhi and Zhang, Yiheng and Pan, Jin and Chrysos, Grigorios G},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

[Add your license here]

## Contact

For questions and issues, please open an issue on GitHub or contact the authors.
