# Retraining the 0.5B models with a corrected gradient

The 0.5B trainer computes its per-token cross-entropy with liger-kernel 0.5.8's fused linear
cross-entropy, whose backward does not apply the per-token loss weights
([Effective gradient](../README.md#effective-gradient)). All released 0.5B checkpoints were trained
that way. To measure how much this matters, we retrained four 0.5B models with
`CDLM_TORCH_CE=chunked`, which computes the same loss with PyTorch's cross-entropy so that autograd
applies every per-token weight. Code generation was evaluated with the released launcher, and CRB with
the authors' frozen copy of the evaluation pipeline (the protocol of `evaluation/crb`, batch size 1).

Apart from the gradient, each run uses the documented recipe of its released counterpart (the
repository launcher: base checkpoint, data, seed 42, 2000 steps, global batch 12, sequence length
4096, learning-rate schedule):

| Name in this directory | Command (run with `CDLM_TORCH_CE=chunked`) | Layout | Released counterpart (Hugging Face, `Shuibai12138/`) |
|---|---|---|---|
| `mdlm_corrected` | `ARM=mdlm bash training/scripts/train_0.5b.sh` | 4 GPUs | `Open-Dcoder-0.5B-baseline-mdm-step2000` (MDLM-0.5B), `mdlm_released` |
| `cdlm_corrected` | `ARM=cdlm bash training/scripts/train_0.5b.sh` | 4 GPUs | `Open-Dcoder-0.5B-mixture-mdm-step2000` (CDLM-0.5B), `cdlm_released` |
| `cdlm_ctw_maxsteps2000_corrected` | `NPROC=2 bash training/scripts/train_clean_token_ablation.sh` | 2 GPUs × accumulation 2 | `open-dcoder-ablation-0.1-ctw0.1`, `cdlm_ctw_maxsteps2000_released` |
| `cdlm_maxsteps2000_corrected` | `NPROC=2 bash training/scripts/train_alpha_sweep.sh 0.1` | 2 GPUs × accumulation 2 | `open-dcoder-ablation-0.1`, `cdlm_maxsteps2000_released` |

- The first two runs see the data stream of the released runs: their step-1 and step-2 losses equal
  those of the released runs to within 1e-6, so the gradient is the only difference.
- The last two used 2 GPUs with gradient accumulation 2 instead of 4 GPUs, at the same global batch,
  so their data stream differs from that of their released counterparts. They are directly
  comparable with each other: their configurations differ only in `clean_token_wt`. The released
  `open-dcoder-ablation-0.1` was trained on another machine (4 × H200) whose shard reading order was
  not recorded.
- Each model is a single training run (seed 42). The retrained checkpoints are not uploaded.

## CRB

`n_replace` = 1, τ = 0.9, macro average over the 12 dataset × error-type cells, all models on the
same pinned instance files (3,213 test-failing programs). Pass@1 after T refinement steps; the
confidence gap and Top-K hit rates are read at the first forward pass.

| Model | Pass@1 T=1 | T=2 | T=3 | T=4 | Gap | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|---|---|---|
| MDLM-0.5B (released) | 0.1386 | 0.2354 | 0.2458 | 0.2445 | 0.0976 | 0.1470 | 0.3427 | 0.5028 |
| `mdlm_corrected` | 0.1055 | 0.1674 | 0.1753 | 0.1754 | 0.0803 | 0.1367 | 0.3229 | 0.4623 |
| CDLM-0.5B (released) | 0.1888 | 0.2789 | 0.2814 | 0.2853 | 0.1636 | 0.2277 | 0.4499 | 0.6058 |
| `cdlm_corrected` | 0.1929 | 0.2512 | 0.2558 | 0.2497 | 0.1897 | 0.2530 | 0.4545 | 0.5879 |
| `open-dcoder-ablation-0.1-ctw0.1` (released) | 0.2038 | 0.1924 | 0.1841 | 0.1817 | 0.1693 | 0.3252 | 0.5333 | 0.6380 |
| `cdlm_ctw_maxsteps2000_corrected` | 0.2424 | 0.2420 | 0.2420 | 0.2430 | 0.2582 | 0.3441 | 0.5034 | 0.5838 |
| `open-dcoder-ablation-0.1` (released) | 0.1966 | 0.2125 | 0.2103 | 0.2046 | 0.1682 | 0.2741 | 0.4687 | 0.5900 |
| `cdlm_maxsteps2000_corrected` | 0.3115 | 0.3370 | 0.3380 | 0.3424 | 0.3636 | 0.3880 | 0.5867 | 0.6890 |

The released rows were re-evaluated alongside the retrained models. They differ from the reference
table in [`evaluation/crb/README.md`](../../evaluation/crb/README.md) by at most 0.0009, from one
HumanEval+ program per cell whose test runs close to the 8 s execution timeout.

Paired over the 3,213 programs (pass@1 pooled over programs; exact McNemar test; 95%
paired-bootstrap interval in [`crb_paired.csv`](crb_paired.csv)):

- `cdlm_corrected` vs `mdlm_corrected`: +0.077, +0.076, +0.076, +0.069 at T = 1..4, p ≤ 1.1e-20.
- Corrected vs released, same data stream: `cdlm_corrected` − CDLM-0.5B is 0.000, −0.028, −0.027,
  −0.037 (p = 1, 7.5e-6, 3.8e-5, 2.4e-8); `mdlm_corrected` − MDLM-0.5B is −0.033, −0.066, −0.071,
  −0.067 (p ≤ 4.3e-16).
- `cdlm_maxsteps2000_corrected` vs `cdlm_ctw_maxsteps2000_corrected` (only the clean-token term
  differs): +0.057, +0.089, +0.090, +0.092 at T = 1..4, p ≤ 5.8e-16.
- `cdlm_corrected` vs `cdlm_ctw_maxsteps2000_corrected` (differ in the clean-token term, the
  learning-rate schedule and the GPU layout): −0.056 at T = 1 (p = 2.9e-14); −0.011, −0.006, −0.012
  at T = 2..4 (p ≥ 0.17).

## Code generation

Released code-generation launcher ([`../../evaluation/codegen`](../../evaluation/codegen/README.md)),
HumanEval(+) and MBPP(+), 10 samples per problem (20 for MBPP+ problems 374 and 375, as in the
recorded runs), pass@k × 100. The released and base values are the recorded runs in
`evaluation/codegen/reference/recorded_runs.json`.

| Decoder | Benchmark | k | Base | MDLM-0.5B | `mdlm_corrected` | CDLM-0.5B | `cdlm_corrected` |
|---|---|---|---|---|---|---|---|
| vanilla | HumanEval | 1 | 18.84 | 19.02 | 20.30 | 21.59 | 22.07 |
| vanilla | HumanEval | 10 | 36.59 | 38.41 | 37.80 | 41.46 | 40.85 |
| vanilla | HumanEval+ | 1 | 16.71 | 16.95 | 17.56 | 20.55 | 19.57 |
| vanilla | HumanEval+ | 10 | 31.71 | 34.15 | 34.15 | 39.02 | 35.37 |
| vanilla | MBPP | 1 | 15.74 | 10.62 | 12.64 | 13.10 | 12.72 |
| vanilla | MBPP | 10 | 35.00 | 31.20 | 31.80 | 35.20 | 31.20 |
| vanilla | MBPP+ | 1 | 23.11 | 17.71 | 18.13 | 20.73 | 19.09 |
| vanilla | MBPP+ | 10 | 50.24 | 44.91 | 43.36 | 47.22 | 41.87 |
| ReMDM | HumanEval | 1 | 18.23 | 19.82 | 19.39 | 21.16 | 21.59 |
| ReMDM | HumanEval | 10 | 32.93 | 38.41 | 36.59 | 42.68 | 38.41 |
| ReMDM | HumanEval+ | 1 | 16.40 | 17.20 | 16.40 | 20.00 | 19.33 |
| ReMDM | HumanEval+ | 10 | 29.27 | 34.15 | 32.32 | 37.80 | 33.54 |
| ReMDM | MBPP | 1 | 16.00 | 10.84 | 11.44 | 14.14 | 12.26 |
| ReMDM | MBPP | 10 | 36.60 | 32.80 | 32.00 | 35.00 | 31.00 |
| ReMDM | MBPP+ | 1 | 23.19 | 16.97 | 18.73 | 21.06 | 18.76 |
| ReMDM | MBPP+ | 10 | 48.54 | 44.42 | 44.02 | 47.60 | 42.96 |

Paired tests (bootstrap interval and sign-flip p over problems) for every pair of models are in
[`codegen_paired.csv`](codegen_paired.csv).

- `cdlm_corrected` scores above `mdlm_corrected` on HumanEval and HumanEval+ in all 8 cells
  (significant only for ReMDM HumanEval+ pass@1, p = 0.012); on MBPP and MBPP+ the two are within
  noise (p ≥ 0.40 for pass@1).
- Both corrected models are below the base model in all 8 MBPP and MBPP+ cells.
- HumanEval with the vanilla decoder only, for the two models trained with the 2000-step schedule:
  `cdlm_ctw_maxsteps2000_corrected` 15.67 / 34.15 and `cdlm_maxsteps2000_corrected` 15.91 / 29.88
  (pass@1 / pass@10), both below the base model (18.84 / 36.59) and below `mdlm_corrected` and
  `cdlm_corrected` in pass@1 (by 4.4 to 6.4 points, p 0.008 to 0.04).

HumanEval+/94 (`cdlm_corrected`, ReMDM): one sample runs close to the 3 s execution timeout. The table
keeps the stored score (0/10); rescoring it (5 of 5 runs) gives 1/10 (pass@1 19.39, pass@10 34.15).
No conclusion changes.

## Files

- [`crb_macro.csv`](crb_macro.csv), [`crb_per_cell.csv`](crb_per_cell.csv),
  [`crb_paired.csv`](crb_paired.csv): CRB results per model, per cell and paired tests (pass@1
  columns of the paired file are pooled over programs).
- [`codegen_passk.csv`](codegen_passk.csv), [`codegen_paired.csv`](codegen_paired.csv): code
  generation.
