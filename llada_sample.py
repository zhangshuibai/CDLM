"""
LLaDA Sampling Algorithms
Implements remask-based sampling strategies for LLaDA models.
"""

import math
import torch
import torch.distributions as dists
import torch.nn.functional as F
from typing import Optional, Dict, Union, List


# ========================================================================
# Helper Functions
# ========================================================================

def top_p_logits(logits, top_p):
    """Apply nucleus (top-p) filtering."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits, top_k):
    """Apply top-k filtering."""
    if top_k is None or top_k == 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(
        indices_to_remove, torch.finfo(logits.dtype).min
    )


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    """Sample tokens from logits."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
    else:
        _, x0 = probs.max(dim=-1)
    return x0


def sample_tokens_with_confidence(
    logits, temperature=0.0, top_p=None, top_k=None, confidence_type="vanilla"
):
    """Sample tokens and compute confidence."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
    else:
        _, x0 = probs.max(dim=-1)

    if confidence_type == "vanilla":
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    elif confidence_type == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif confidence_type == "entropy":
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)
    else:
        raise ValueError(f"Unknown confidence type: {confidence_type}")

    return confidence, x0


def mad_outlier_detection(values, k=2.5):
    """
    MAD (Median Absolute Deviation) based outlier detection.
    
    Args:
        values: [batch_size, seq_len] tensor of confidence values
        k: threshold multiplier (default 2.5, typical values: 2.5 or 3.0)
    
    Returns:
        is_outlier: [batch_size, seq_len] boolean tensor indicating outliers
    """
    batch_size, seq_len = values.shape
    is_outlier = torch.zeros_like(values, dtype=torch.bool)
    
    # Process each sample independently
    for i in range(batch_size):
        sample_values = values[i]
        
        # Skip if no valid values
        if sample_values.numel() == 0:
            continue
        
        # Compute median
        median_val = torch.median(sample_values)
        
        # Compute absolute deviations from median
        abs_deviations = torch.abs(sample_values - median_val)
        
        # Compute MAD (median of absolute deviations)
        mad = torch.median(abs_deviations)
        
        # Handle edge case: if MAD is 0, use a small epsilon
        if mad < 1e-10:
            mad = torch.tensor(1e-10, device=mad.device, dtype=mad.dtype)
        
        # Use MADe (normalized MAD) for consistency with standard deviation
        # MADe = 1.4826 * MAD (for normal distribution)
        made = 1.4826 * mad
        
        # Mark outliers: values that deviate more than k * MADe from median
        threshold = k * made
        is_outlier[i] = abs_deviations > threshold
    
    return is_outlier


def find_outlier_tokens(conf_full, variable_mask, k=2.5):
    """
    Find outlier tokens using MAD (Median Absolute Deviation) algorithm.
    Only considers variable positions (per-sample).
    
    Args:
        conf_full: [batch_size, seq_len] tensor of confidence values
        variable_mask: [batch_size, seq_len] boolean tensor indicating variable positions
        k: threshold multiplier for MAD (default 2.5, typical values: 2.5 or 3.0)
    
    Returns:
        is_outlier: [batch_size, seq_len] boolean tensor indicating outlier tokens
    """
    batch_size, seq_len = conf_full.shape
    is_outlier = torch.zeros_like(conf_full, dtype=torch.bool)
    
    # Process each sample independently
    for i in range(batch_size):
        # Extract variable positions' confidence for this sample
        sample_variable_mask = variable_mask[i]
        if not sample_variable_mask.any():
            continue
        
        sample_conf = conf_full[i]
        variable_conf = sample_conf[sample_variable_mask]
        
        if variable_conf.numel() == 0:
            continue
        
        # Compute median of variable confidences
        median_val = torch.median(variable_conf)
        
        # Compute absolute deviations from median
        abs_deviations = torch.abs(variable_conf - median_val)
        
        # Compute MAD (median of absolute deviations)
        mad = torch.median(abs_deviations)
        
        # Handle edge case: if MAD is 0, use a small epsilon
        if mad < 1e-10:
            mad = torch.tensor(1e-10, device=mad.device, dtype=mad.dtype)
        
        # Use MADe (normalized MAD) for consistency with standard deviation
        made = 1.4826 * mad
        
        # Mark LOW-confidence outliers:
        # values significantly BELOW the median by more than k * MADe
        threshold = k * made
        sample_is_outlier = (variable_conf < median_val) & (abs_deviations > threshold)
        
        # Map back to original positions
        variable_positions = sample_variable_mask.nonzero(as_tuple=True)[0]
        is_outlier[i, variable_positions] = sample_is_outlier
    
    return is_outlier


def mad_outlier_detection(values, k=2.5):
    """
    MAD (Median Absolute Deviation) based outlier detection.
    
    Args:
        values: [batch_size, seq_len] tensor of confidence values
        k: threshold multiplier (default 2.5, typical values: 2.5 or 3.0)
    
    Returns:
        is_outlier: [batch_size, seq_len] boolean tensor indicating outliers
    """
    batch_size, seq_len = values.shape
    is_outlier = torch.zeros_like(values, dtype=torch.bool)
    
    # Process each sample independently
    for i in range(batch_size):
        sample_values = values[i]
        
        # Skip if no valid values
        if sample_values.numel() == 0:
            continue
        
        # Compute median
        median_val = torch.median(sample_values)
        
        # Compute absolute deviations from median
        abs_deviations = torch.abs(sample_values - median_val)
        
        # Compute MAD (median of absolute deviations)
        mad = torch.median(abs_deviations)
        
        # Handle edge case: if MAD is 0, use a small epsilon
        if mad < 1e-10:
            mad = torch.tensor(1e-10, device=mad.device, dtype=mad.dtype)
        
        # Use MADe (normalized MAD) for consistency with standard deviation
        # MADe = 1.4826 * MAD (for normal distribution)
        made = 1.4826 * mad
        
        # Mark outliers: values that deviate more than k * MADe from median
        threshold = k * made
        is_outlier[i] = abs_deviations > threshold
    
    return is_outlier


def find_outlier_tokens(conf_full, variable_mask, k=2.5):
    """
    Find outlier tokens using MAD (Median Absolute Deviation) algorithm.
    Only considers variable positions (per-sample).
    
    Args:
        conf_full: [batch_size, seq_len] tensor of confidence values
        variable_mask: [batch_size, seq_len] boolean tensor indicating variable positions
        k: threshold multiplier for MAD (default 2.5, typical values: 2.5 or 3.0)
    
    Returns:
        is_outlier: [batch_size, seq_len] boolean tensor indicating outlier tokens
    """
    batch_size, seq_len = conf_full.shape
    is_outlier = torch.zeros_like(conf_full, dtype=torch.bool)
    
    # Process each sample independently
    for i in range(batch_size):
        # Extract variable positions' confidence for this sample
        sample_variable_mask = variable_mask[i]
        if not sample_variable_mask.any():
            continue
        
        sample_conf = conf_full[i]
        variable_conf = sample_conf[sample_variable_mask]
        
        if variable_conf.numel() == 0:
            continue
        
        # Compute median of variable confidences
        median_val = torch.median(variable_conf)
        
        # Compute absolute deviations from median
        abs_deviations = torch.abs(variable_conf - median_val)
        
        # Compute MAD (median of absolute deviations)
        mad = torch.median(abs_deviations)
        
        # Handle edge case: if MAD is 0, use a small epsilon
        if mad < 1e-10:
            mad = torch.tensor(1e-10, device=mad.device, dtype=mad.dtype)
        
        # Use MADe (normalized MAD) for consistency with standard deviation
        made = 1.4826 * mad
        
        # Mark LOW-confidence outliers:
        # values significantly BELOW the median by more than k * MADe
        threshold = k * made
        sample_is_outlier = (variable_conf < median_val) & (abs_deviations > threshold)
        
        # Map back to original positions
        variable_positions = sample_variable_mask.nonzero(as_tuple=True)[0]
        is_outlier[i, variable_positions] = sample_is_outlier
    
    return is_outlier


# ========================================================================
# External AR Model Confidence
# ========================================================================

_external_ar_model_cache = {}


def get_external_AR_conf(
    tokens,
    model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"
):
    """Compute confidence using external AR model."""
    global _external_ar_model_cache

    if model_name not in _external_ar_model_cache:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        _external_ar_model_cache[model_name] = model
        print(f"Loaded external AR model: {model_name}")

    model = _external_ar_model_cache[model_name]
    device = tokens.device
    batch_size, seq_len = tokens.shape

    with torch.no_grad():
        model_device = next(model.parameters()).device
        tokens_on_model = tokens.to(model_device)

        outputs = model(input_ids=tokens_on_model)
        logits = outputs.logits

        shifted_logits = logits[:, :-1, :]
        shifted_tokens = tokens_on_model[:, 1:]

        probs = torch.softmax(shifted_logits.float(), dim=-1)
        confidence = torch.gather(
            probs, dim=-1, index=shifted_tokens.unsqueeze(-1)
        ).squeeze(-1)

        first_token_conf = torch.ones(
            batch_size, 1, device=confidence.device, dtype=confidence.dtype
        )
        confidence = torch.cat([first_token_conf, confidence], dim=1)
        confidence = confidence.to(device)

    return confidence


# ========================================================================
# Remask Scheduling
# ========================================================================

def compute_remask_schedule(
    step, total_steps, num_variable_positions, schedule_type="linear"
):
    """Compute number of tokens to remask at current step."""
    if schedule_type == "linear":
        remask_ratio = (total_steps - step - 1) / total_steps
    elif schedule_type == "cosine":
        remask_ratio = 0.5 * (
            1 + math.cos(math.pi * (step + 1) / total_steps)
        )
    elif schedule_type.startswith("fixed_"):
        ratio_str = schedule_type.replace("fixed_", "")
        remask_ratio = float(ratio_str)
        if not (0.0 <= remask_ratio <= 1.0):
            raise ValueError(
                f"Fixed remask ratio must be in [0, 1], got {remask_ratio}"
            )
    else:
        raise NotImplementedError(
            f"Schedule type {schedule_type} not implemented"
        )

    num_to_remask = (
        num_variable_positions.float() * remask_ratio
    ).floor().to(torch.long)
    return num_to_remask.clamp_min(0)


# ========================================================================
# Helper function for getting logits with attention_mask support
# ========================================================================

def get_model_logits(model, input_ids, attention_mask=None, is_causal=None):
    """
    Get logits from model with optional attention_mask and is_causal flag.
    
    Args:
        model: The model to get logits from
        input_ids: Input token IDs
        attention_mask: Optional attention mask (bool or long tensor)
        is_causal: Optional flag for causal (True) vs bidirectional (False) attention
    """
    # Build kwargs for model forward pass
    kwargs = {}
    if attention_mask is not None and not attention_mask.all():
        # Convert bool attention_mask to long (0/1) as expected by transformers
        if attention_mask.dtype == torch.bool:
            kwargs['attention_mask'] = attention_mask.long()
        else:
            kwargs['attention_mask'] = attention_mask
    if is_causal is not None:
        kwargs['is_causal'] = is_causal
    
    if kwargs:
        return model(input_ids, **kwargs).logits
    else:
        return model(input_ids).logits


# ========================================================================
# Main Sampling Function
# ========================================================================

@torch.no_grad()
def llada_sample(
    model,
    input_ids: torch.LongTensor,
    fix_mask: torch.BoolTensor,
    mask_id: int,
    attention_mask: Optional[torch.BoolTensor] = None,
    steps: int = 10,
    algorithm: str = "random-remask",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    remask_scheduler: str = "linear",
    confidence_threshold: Optional[float] = None,
    external_ar_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    return_history: bool = False,
    model_name: Optional[str] = None,
    mad_k: float = 2.5,
) -> Dict:
    """
    Generate text using LLaDA with remask-based sampling.

    Args:
        model: LLaDA model
        input_ids: [batch_size, seq_len] token IDs (with masks already)
        fix_mask: [batch_size, seq_len] True=fixed, False=variable
        mask_id: mask token ID
        attention_mask: [batch_size, seq_len] attention mask (1=attend,
                       0=ignore). If None, all positions attend.
        steps: number of generation steps
        algorithm: 'random-remask', 'external-AR_conf-remask',
                   'self_conf-remask:vanilla', 'self_conf-remask:entropy',
                   'self_conf-remask:topk_margin', 'self_conf-remask:vanilla_MAD',
                   'left2right_AR'
        temperature: sampling temperature (0.0 = greedy)
        top_p: nucleus sampling threshold
        top_k: top-k sampling threshold
        remask_scheduler: 'linear', 'cosine', or 'fixed_X.X'
        confidence_threshold: optional float. If set, remask every variable
            position whose confidence is below this threshold, ignoring the
            scheduler/top-k logic.
        external_ar_model_name: external AR model name
        return_history: whether to return generation history
        model_name: optional model name for special handling (e.g., Dream-v0 
                   models require logits shifting)
        mad_k: threshold multiplier for MAD algorithm (default 2.5, typical 
               values: 2.0, 2.5, 3.0). Only used when algorithm is 
               'self_conf-remask:vanilla_MAD'

    Returns:
        dict with 'sequences', 'actual_steps', and optionally 'history'
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    x = input_ids.clone()

    # Handle -1 as a special value meaning "no confidence threshold"
    # Convert -1 to None to use regular scheduler logic
    if confidence_threshold == -1:
        confidence_threshold = None

    # Validate mutual exclusivity between confidence_threshold and MAD algorithm
    is_mad_algorithm = algorithm == "self_conf-remask:vanilla_MAD"
    if confidence_threshold is not None and is_mad_algorithm:
        raise ValueError(
            "confidence_threshold and MAD algorithm (self_conf-remask:vanilla_MAD) "
            "are mutually exclusive. Please use only one of them."
        )

    # Create default attention_mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(x, dtype=torch.bool)

    # Initialize history
    if return_history:
        history = {
            'prompt': {
                'fix_mask': fix_mask.clone().cpu(),
                'tokens': x.clone().cpu()
            },
            'steps': []
        }
    else:
        history = None

    # Track variable positions (constant throughout the loop since fix_mask doesn't change)
    variable_mask = ~fix_mask
    has_variable_positions = variable_mask.any(dim=1)

    # Initialize actual_steps: track per-sample if confidence_threshold is set
    # or if using MAD-based algorithm (they are mutually exclusive)
    is_mad_algorithm = algorithm == "self_conf-remask:vanilla_MAD"
    use_per_sample_tracking = (
        confidence_threshold is not None or is_mad_algorithm
    )
    if use_per_sample_tracking:
        actual_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        actual_steps = torch.full(
            (batch_size,), steps, dtype=torch.long, device=device
        )
        done_mask = None
    # Track variable positions (constant throughout the loop since fix_mask doesn't change)
    variable_mask = ~fix_mask
    has_variable_positions = variable_mask.any(dim=1)

    # Initialize actual_steps: track per-sample if confidence_threshold is set
    # or if using MAD-based algorithm (they are mutually exclusive)
    is_mad_algorithm = algorithm == "self_conf-remask:vanilla_MAD"
    use_per_sample_tracking = (
        confidence_threshold is not None or is_mad_algorithm
    )
    if use_per_sample_tracking:
        actual_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        actual_steps = torch.full(
            (batch_size,), steps, dtype=torch.long, device=device
        )
        done_mask = None

    # Generation loop
    for step in range(steps):
        # Get logits with special handling for open-dcoder models
        is_open_dcoder = model_name and "open-dcoder" in model_name.lower()
        
        # For open-dcoder, use bidirectional attention (is_causal=False)
        # IMPORTANT: Don't pass attention_mask with is_causal=False for open-dcoder
        # as it causes NaN issues. Let model attend to all positions (including padding).
        # The fix_mask ensures padding positions are never sampled.
        if is_open_dcoder:
            logits = get_model_logits(model, x, attention_mask=None, is_causal=False)
        else:
            logits = get_model_logits(model, x, attention_mask)
        
        # Special handling for open-dcoder and Dream-v0 models: shift logits RIGHT
        # These models output logits where logits[:, i] predicts token[:, i+1]
        # We need to shift RIGHT so logits[:, i] predicts token[:, i]
        # This matches Open-dLLM's implementation
        if is_open_dcoder or (model_name and "Dream-v0" in model_name):
            # Shift logits right: keep first position, discard last
            # Original: logits[i] -> token[i+1]
            # After shift: logits[i] -> token[i]
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        # Determine current masked positions (to be predicted this step)
        if algorithm == "left2right_AR":
            # Left-to-right AR: find leftmost mask position for each sample
            current_mask = torch.zeros_like(x, dtype=torch.bool)
            for b in range(batch_size):
                mask_positions = (x[b] == mask_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) > 0:
                    current_mask[b, mask_positions[0]] = True
        else:
            # Standard: predict all masked positions
            current_mask = (x == mask_id) & variable_mask

        # Predict ONLY currently masked positions
        if current_mask.any():
            masked_logits = logits[current_mask]
            
            # All algorithms sample tokens the same way
            # (confidence is computed separately later based on algorithm)
            x0_masked = sample_tokens(
                masked_logits, temperature, top_p, top_k
            )
            # Write back predictions only to masked positions
            x[current_mask] = x0_masked
        else:
            # Nothing to predict this step (should be early-exited above)
            x0_masked = torch.empty(0, device=device, dtype=torch.long)

        # Compute remask schedule (based on all variable positions) when needed
        # Skip for left2right_AR (no remasking)
        if algorithm != "left2right_AR" and confidence_threshold is None:
            num_variable_positions = variable_mask.sum(dim=1)
            num_to_remask = compute_remask_schedule(
                step, steps, num_variable_positions, remask_scheduler
            )

        # Compute prob_all from model logits
        probs = torch.softmax(logits.float(), dim=-1)
        prob_all = torch.gather(
            probs, dim=-1, index=x.unsqueeze(-1)
        ).squeeze(-1)

        # Build confidence map over ALL positions (skip for left2right_AR)
        if algorithm != "left2right_AR":
            conf_full = torch.full_like(x, float("inf"), dtype=torch.float32)
        else:
            # Dummy conf_full for left2right_AR (not used)
            conf_full = torch.zeros_like(x, dtype=torch.float32)

        # Compute conf_full based on algorithm
        if algorithm == "left2right_AR":
            # Left-to-right AR: record prob_all as confidence for history
            # (not used for remasking, but useful for analysis)
            conf_full[variable_mask] = prob_all[variable_mask]
        elif algorithm == "external-AR_conf-remask":
            # Use external AR confidence for remask decision
            conf_full[variable_mask] = get_external_AR_conf(
                x, model_name=external_ar_model_name
            )[variable_mask]
        elif algorithm == "random-remask":
            # Random confidence on variable positions
            conf_full[variable_mask] = torch.rand_like(
                prob_all[variable_mask], dtype=torch.float32
            )
        elif algorithm.startswith("self_conf-remask:"):
            # Self-confidence: use model probabilities as confidence
            conf_full[variable_mask] = prob_all[variable_mask]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        to_remask = torch.zeros_like(x, dtype=torch.bool)
        
        # Left-to-right AR: no remasking, just unmask one token at a time
        if algorithm == "left2right_AR":
            # Check if all masks are resolved for early exit
            remaining_masks = (x == mask_id).any(dim=1)
            if not remaining_masks.any():
                # All masks resolved, mark as done and break early
                actual_steps[has_variable_positions] = step + 1
                break
            # Update actual_steps for active samples
            actual_steps[has_variable_positions] = step + 1
        # MAD algorithm and confidence_threshold are mutually exclusive
        # MAD algorithm has priority
        elif algorithm == "self_conf-remask:vanilla_MAD":
            # MAD-based outlier detection for remasking
            # Only remask tokens that are outliers according to MAD
            is_outlier = find_outlier_tokens(conf_full, variable_mask, k=mad_k)
            
            # Only remask outliers that are in variable positions
            to_remask = is_outlier & variable_mask
            
            # Exclude already-done samples
            to_remask = to_remask & (~done_mask.unsqueeze(1))

            # Update actual_steps for active samples
            active_samples = has_variable_positions & (~done_mask)
            actual_steps[active_samples] = step + 1

            # Mark samples as done if no tokens need remasking
            no_remask = ~to_remask.any(dim=1)
            newly_done = no_remask & (~done_mask) & has_variable_positions
            if newly_done.any():
                done_mask[newly_done] = True
        elif confidence_threshold is not None:
            # Confidence threshold-based remasking
            to_remask = (conf_full <= confidence_threshold) & variable_mask
            to_remask = to_remask & (~done_mask.unsqueeze(1))

            # Update actual_steps for active samples
            active_samples = has_variable_positions & (~done_mask)
            actual_steps[active_samples] = step + 1

            # Mark samples as done if no tokens need remasking
            no_remask = ~to_remask.any(dim=1)
            newly_done = no_remask & (~done_mask) & has_variable_positions
            if newly_done.any():
                done_mask[newly_done] = True
            to_remask = to_remask & (~done_mask.unsqueeze(1))

            # Update actual_steps for active samples
            active_samples = has_variable_positions & (~done_mask)
            actual_steps[active_samples] = step + 1

            # Mark samples as done if no tokens need remasking
            no_remask = ~to_remask.any(dim=1)
            newly_done = no_remask & (~done_mask) & has_variable_positions
            if newly_done.any():
                done_mask[newly_done] = True
        else:
            sorted_idx = torch.argsort(conf_full, dim=1, descending=False)
            max_k = int(num_to_remask.max().item())

            if max_k > 0:
                topk_idx = sorted_idx[:, :max_k]
                row_mask = (
                    torch.arange(max_k, device=device).unsqueeze(0) <
                    num_to_remask.unsqueeze(1)
                )

                batch_arange = torch.arange(
                    batch_size, device=device
                ).unsqueeze(1).expand_as(topk_idx)
                valid_batch = batch_arange[row_mask]
                valid_col = topk_idx[row_mask]
                to_remask[valid_batch, valid_col] = True

        # Apply remasking (not on last step, and skip for left2right_AR)
        if step < steps - 1 and algorithm != "left2right_AR":
            x[to_remask] = mask_id

        # Record history
        if history is not None:
            step_history = {
                'x0_variable': x[variable_mask].clone().cpu(),
                'conf_variable': conf_full[variable_mask].clone().cpu(),
                'prob_variable': prob_all[variable_mask].clone().cpu(),
                'remask_positions': to_remask.clone().cpu(),
                'sequence': x.clone().cpu(),
            }
            history['steps'].append(step_history)

    result = {'sequences': x, 'actual_steps': actual_steps}
    if return_history:
        result['history'] = history

    return result


# ========================================================================
# Convenience Function with Preprocessing
# ========================================================================

def llada_sample_text(
    model,
    tokenizer,
    prompt: Union[str, List[str]],
    max_new_tokens: int = 50,
    mask_id: int = 126336,
    pad_id: Optional[int] = None,
    steps: int = 10,
    **kwargs
) -> Dict:
    """
    Generate text from string prompt(s) with preprocessing.

    Args:
        model: LLaDA model
        tokenizer: tokenizer
        prompt: input text prompt (str or List[str] for batch)
        max_new_tokens: maximum number of tokens to generate
        mask_id: mask token ID
        pad_id: pad token ID (auto-determined if batch input)
        **kwargs: additional arguments for llada_sample

    Returns:
        dict with 'text', 'sequences', 'actual_steps', and optional 'history'
    """
    device = next(model.parameters()).device

    # Handle batch input
    is_batch = isinstance(prompt, list)
    if is_batch:
        prompts = prompt
    else:
        prompts = [prompt]

    # Get pad_id if batch mode
    if is_batch and pad_id is None:
        has_pad = (
            hasattr(tokenizer, 'pad_token_id') and
            tokenizer.pad_token_id is not None
        )
        if has_pad:
            pad_id = tokenizer.pad_token_id
        else:
            raise ValueError(
                "Batch mode requires pad_id, but tokenizer has no "
                "pad_token_id. Please provide pad_id."
            )

    # Validate pad_id
    if pad_id is not None and pad_id == mask_id:
        raise ValueError(
            f"pad_id ({pad_id}) cannot be the same as "
            f"mask_id ({mask_id}). Padding tokens should not "
            "be predicted by the model."
        )

    # Tokenize prompts
    tokenized = [
        tokenizer(p, return_tensors="pt")["input_ids"]
        for p in prompts
    ]

    # For each prompt: [prompt_tokens] + [mask_ids] (+ [pad_ids] if needed)
    input_ids_list = []
    fix_mask_list = []
    attention_mask_list = []

    for tokens in tokenized:
        plen = tokens.shape[1]
        # Add masks after prompt
        masks = torch.full((1, max_new_tokens), mask_id, dtype=torch.long)
        seq = torch.cat([tokens, masks], dim=1)  # [1, plen + max_new_tokens]

        # Create masks for this sequence
        # fix_mask: only prompt tokens are fixed
        fix = torch.zeros(1, seq.shape[1], dtype=torch.bool)
        fix[0, :plen] = True

        # attention_mask: all tokens are valid (no padding yet)
        attn = torch.ones(1, seq.shape[1], dtype=torch.bool)

        input_ids_list.append(seq)
        fix_mask_list.append(fix)
        attention_mask_list.append(attn)

    # Find max length and pad all sequences (LEFT padding for open-dcoder compatibility)
    max_len = max(seq.shape[1] for seq in input_ids_list)

    # Pad sequences to max_len
    for i in range(len(input_ids_list)):
        current_len = input_ids_list[i].shape[1]
        if current_len < max_len:
            pad_len = max_len - current_len
            # LEFT padding: add padding tokens at the beginning
            padding = torch.full(
                (1, pad_len), pad_id, dtype=torch.long
            )
            input_ids_list[i] = torch.cat(
                [padding, input_ids_list[i]], dim=1
            )
            # Pad fix_mask (padding is fixed)
            pad_fix = torch.ones((1, pad_len), dtype=torch.bool)
            fix_mask_list[i] = torch.cat(
                [pad_fix, fix_mask_list[i]], dim=1
            )
            # Pad attention_mask (padding is not attended)
            pad_attn = torch.zeros((1, pad_len), dtype=torch.bool)
            attention_mask_list[i] = torch.cat(
                [pad_attn, attention_mask_list[i]], dim=1
            )

    # Concatenate into batches
    input_ids = torch.cat(input_ids_list, dim=0).to(device)
    fix_mask = torch.cat(fix_mask_list, dim=0).to(device)
    attention_mask = torch.cat(attention_mask_list, dim=0).to(device)

    # Generate
    result = llada_sample(
        model=model,
        input_ids=input_ids,
        fix_mask=fix_mask,
        mask_id=mask_id,
        attention_mask=attention_mask,
        steps=steps,
        **kwargs
    )

    # Decode
    sequences = result['sequences']
    text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    result['text'] = text

    return result


# ========================================================================
# Example Usage
# ========================================================================

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    from transformers import AutoModel, AutoTokenizer

    device = 'cuda'
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )

    # Single prompt example
    print("=" * 60)
    print("Single prompt example")
    print("=" * 60)
    prompt = "The capital of France is"
    result = llada_sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        steps=10,
        algorithm='self_conf-remask:vanilla',
        temperature=0.0,
        remask_scheduler="linear",
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {result['text'][0]}")
    print(f"Steps: {result['actual_steps'][0].item()}\n")

    # Batch prompts example
    print("=" * 60)
    print("Batch prompts example")
    print("=" * 60)
    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "In machine learning,",
    ]
    result = llada_sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompts,  # Pass list of strings
        max_new_tokens=15,
        steps=10,
        algorithm='self_conf-remask:vanilla',
        temperature=0.0,
        remask_scheduler="linear",
    )
    for i, (p, text, steps) in enumerate(
        zip(prompts, result['text'], result['actual_steps'])
    ):
        print(f"[{i}] Prompt: {p}")
        print(f"    Generated: {text}")
        print(f"    Steps: {steps.item()}\n")
