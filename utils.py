"""Utility functions for model loading and tokenization."""
import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device: str = None, local_rank: int = None):
    """
    Load model and tokenizer with appropriate settings based on model type.
    
    Args:
        model_name: Name or path of the model
        device: Device to load model on (default: "cuda:{local_rank}" or "cuda:0")
        local_rank: Local rank for distributed training (default: from LOCAL_RANK env var or 0)
    
    Returns:
        tuple: (model, tokenizer, pad_id, mask_id)
            - model: Loaded model
            - tokenizer: Loaded tokenizer
            - pad_id: Padding token ID
            - mask_id: Mask token ID
    """
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if device is None:
        device = f"cuda:{local_rank}"
    
    print(f"[Rank {local_rank}] Loading model...")
    
    # Check if this is an open-dcoder model
    is_open_dcoder = "open-dcoder" in model_name.lower()
    
    if is_open_dcoder:
        print(f"[Rank {local_rank}] Detected open-dcoder model, using custom Qwen2ForCausalLM...")
        from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False  # Fix for tokenizer format compatibility
        )
        
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()
        
        print(f"[Rank {local_rank}] Successfully loaded custom Qwen2ForCausalLM")
    else:
        # Standard model loading
        print(f"[Rank {local_rank}] Loading model with AutoModel...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False  # Fix for tokenizer format compatibility
        )

    # Get pad_id
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.eos_token_id
        print(f"Warning: Using eos_token_id as pad_id: {pad_id}")
    
    # Determine mask_id: LLaDA uses 126336 (<|mdm_mask|> from additional_special_tokens),
    # open-dcoder uses 151665, others use tokenizer's mask_token_id
    if "LLaDA" in model_name or "llada" in model_name.lower():
        mask_id = 126336  # <|mdm_mask|> token ID for LLaDA models
        print(f"[Rank {local_rank}] Using LLaDA mask_id: {mask_id} (<|mdm_mask|>)")
    elif is_open_dcoder:
        mask_id = 151665  # Hard-coded mask token ID for open-dcoder models
        print(f"[Rank {local_rank}] Using open-dcoder mask_id: {mask_id}")
    else:
        if tokenizer.mask_token_id is not None:
            mask_id = tokenizer.mask_token_id
            print(f"[Rank {local_rank}] Using tokenizer's mask_token_id: {mask_id}")
        else:
            raise ValueError(f"Mask token not found in tokenizer for model: {model_name}")
    
    return model, tokenizer, pad_id, mask_id


def build_output_paths(
    initial_results_file: str,
    refined_steps: int,
    refine_setting: str,
    algorithm: str,
    local_rank: int = 0,
    world_size: int = 1,
    output_prefix: str = "correction",
    confidence_threshold: float = None,
    mad_k: float = None,
    temperature: float = None
):
    """
    Build output paths for refined results and history by reusing input file path structure.
    
    Args:
        initial_results_file: Path to input JSONL file (e.g., codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1.jsonl)
        refined_steps: Number of refinement steps
        refine_setting: Refinement setting (e.g., "remove_all")
        algorithm: Algorithm name (e.g., "self_conf-remask:vanilla")
        local_rank: Local rank for distributed training (default: 0)
        world_size: World size for distributed training (default: 1)
        output_prefix: Prefix for output directories (default: "correction")
    
    Returns:
        dict: Dictionary containing paths with keys:
            - output_file: Path to merged output JSONL file (str)
            - output_dir: Output directory Path object
            - history_dir: History directory Path object
            - history_file: History file Path object
    
    Example:
        Input: codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1.jsonl
        Output structure (with default prefix="correction"):
        - correction_results/refined_steps{steps}/{setting}/{algorithm}/codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1/{file}
        - correction_history/refined_steps{steps}/{setting}/{algorithm}/codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1/{file}
        
        With custom prefix="AR_correction":
        - AR_correction_results/refined_steps{steps}/{setting}/{algorithm}/codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1/{file}
        - AR_correction_history/refined_steps{steps}/{setting}/{algorithm}/codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1/{file}
    """
    # Build output paths by reusing input file path structure
    # Input: codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1.jsonl
    # Output: correction_results/refined_steps{steps}/{setting}/{algorithm}/codecorrection/human-eval/LLaDA-8B-Base_unary_wrong_1/{file}
    input_path = Path(initial_results_file)
    # Check if path has a file extension to determine if it's a file or directory
    is_file_path = input_path.suffix in ['.jsonl', '.json', '.pt', '.txt'] or input_path.is_file()
    input_dir = input_path.parent if is_file_path else input_path
    input_stem = input_path.stem if is_file_path else input_path.name
    
    # Normalize algorithm name for filesystem (replace ':' with '_')
    algorithm_safe = algorithm.replace(':', '_')
    
    # Build algorithm suffix with parameters
    algorithm_suffix = algorithm_safe
    if algorithm == "self_conf-remask:vanilla" and confidence_threshold is not None:
        # Format: self_conf-remask_vanilla_ct0.9
        ct_str = f"{confidence_threshold:.2f}".replace('.', '')
        algorithm_suffix = f"{algorithm_safe}_ct{ct_str}"
    elif algorithm == "self_conf-remask:vanilla_MAD" and mad_k is not None:
        # Format: self_conf-remask_vanilla_MAD_k2.5
        k_str = f"{mad_k:.1f}".replace('.', '')
        algorithm_suffix = f"{algorithm_safe}_k{k_str}"
    
    # Add temperature if specified (always include, including 0)
    if temperature is not None:
        temp_str = f"{temperature:.1f}".replace('.', '')
        algorithm_suffix = f"{algorithm_suffix}_t{temp_str}"
    
    # Build base directories: separate results and history
    results_base_dir = (
        Path(f"{output_prefix}_results") /
        f"refined_steps{refined_steps}" / refine_setting / algorithm_suffix
    )
    history_base_dir = (
        Path(f"{output_prefix}_history") /
        f"refined_steps{refined_steps}" / refine_setting / algorithm_suffix
    )
    
    # Reuse input path structure: base_dir/{input_dir}/{input_stem}/
    output_dir = results_base_dir / input_dir / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file name
    output_file = str(output_dir / f"{input_stem}_results_refined.jsonl")
    
    # History directory (separate from results)
    history_dir = history_base_dir / input_dir / input_stem
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # History file path (rank-specific for distributed training)
    if world_size > 1:
        history_file = history_dir / f"refined_rank{local_rank}.pt"
    else:
        history_file = history_dir / "refined_rank0.pt"
    
    return {
        'output_file': output_file,
        'output_dir': output_dir,
        'history_dir': history_dir,
        'history_file': history_file,
    }

