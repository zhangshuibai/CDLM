# veomni/models/transformers/qwen2/generation_utils.py

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    if top_k is None or top_k == 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, alg="origin"):
    # original_dtype = logits.dtype
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
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)
    elif alg in ["maskgit_plus", "origin", "p2"]:
        pass
    else:
        raise NotImplementedError(f"Algorithm {alg} not implemented.")
    
    return confidence, x0


@dataclass
class MDMModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None

class MDMGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        # Set do_sample=True as default for MDM (since MDM handles its own sampling)
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True
        
        super().__init__(**kwargs)
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'entropy')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", 0.0)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)


class MDMGenerationMixin:
    """
    Mixin class for Masked Diffusion Model generation, adapted from the Dream model's generation utils.
    """
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _mdm_prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs
    ) -> MDMGenerationConfig:
        if generation_config is None:
            generation_config = self.generation_config
        
        # Use MDMGenerationConfig as the target class
        if not isinstance(generation_config, MDMGenerationConfig):
            generation_config = MDMGenerationConfig.from_dict(generation_config.to_dict())

        # Update with kwargs
        generation_config.update(**kwargs)
        return generation_config

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MDMGenerationConfig] = None,
        **kwargs,
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # 1. Prepare generation config
        generation_config = self._mdm_prepare_generation_config(generation_config, **kwargs)

        # 2. Prepare inputs
        input_ids = inputs
        attention_mask = kwargs.get("attention_mask", None)

        if input_ids is None:
            raise ValueError("`inputs` must be provided for diffusion generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = input_ids.shape[-1] + generation_config.max_new_tokens
        
        # 3. Expand inputs for multi-sequence generation
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")
        
        input_ids = F.pad(input_ids, (0, generation_config.max_length - input_ids.shape[1]), value=generation_config.mask_token_id)
        attention_mask = None

        # 4. Run the sampling loop
        return self._mdm_sample(
            x=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    def _mdm_sample(
        self,
        x: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MDMGenerationConfig
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # Extract params from config

        # import pdb; pdb.set_trace()
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")

        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if generation_config.output_history else None

        # Pad input_ids to max_length with mask tokens
        # x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # Fixed tokens = input context (should never be remasked in p2)
        fix_mask = (x != mask_token_id)
        # fix_mask = F.pad(fix_mask, (0, max_length - fix_mask.shape[1]), value=0)

        # The model expects a bidirectional mask, so we just use the presence of pad_token_id
        gen_attention_mask = (x != self.config.pad_token_id).long() if self.config.pad_token_id is not None else None

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        # Initialize confidence_history for p2_upgraded_ReMDM
        # -inf means "not yet unmasked", will be updated when token is first unmasked
        if alg == "p2_upgraded_ReMDM":
            confidence_history = torch.full_like(x, float("-inf"), dtype=torch.float32, device=x.device)
            # Fixed positions should never be selected for remasking
            confidence_history[fix_mask] = float("inf")

        for i in range(steps):
            mask_index = (x == mask_token_id)
            if not mask_index.any(): # Stop if no tokens are masked
                break

            # is_causal=False is crucial for bidirectional attention
            outputs = self(input_ids=x, attention_mask=gen_attention_mask, is_causal=False)
            logits = outputs.logits

            # CRITICAL: Shift logits to predict the next token, aligning with training
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.full_like(x[mask_index], fill_value=mask_token_id, device=self.device, dtype=torch.long)
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                x0[transfer_index_t_s] = sampled_tokens
                x[mask_index] = x0

            elif alg == "p2":
                # Use sample_tokens to obtain confidence and candidate tokens for the whole sequence
                # kappa_t: fraction of tokens to keep unmasked (can be replaced with custom schedule)
                kappa_t = (i + 1) / steps

                # Compute confidence and sampled tokens for the entire sequence:
                #   conf_full: [B, L], confidence of the sampled token at each position
                #   x0_full:  [B, L], sampled token IDs for each position
                conf_full, x0_full = sample_tokens(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                )

                # Construct full_conf matrix and mask out fixed positions
                # Only positions in (~fix_mask) are candidates for masking/unmasking
                full_conf = conf_full.clone()
                full_conf[fix_mask] = float("inf")
                # Prevent NaNs or extreme values from interfering
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )

                # Calculate how many positions to re-mask per sample
                # = number of variable positions * (1 - kappa_t)
                num_positions = (~fix_mask).sum(dim=1)  # [B]
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                # Boundaries: at least 0, at most total number of variable positions
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)

                # Select the lowest-confidence positions within (~fix_mask) for re-masking
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                max_k = int(num_to_mask.max().item())
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                    row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                    to_mask = torch.zeros_like(x, dtype=torch.bool)
                    batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                    valid_batch = batch_arange[row_mask]  # [sum_k]
                    valid_col   = topk_idx[row_mask]      # [sum_k]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x, dtype=torch.bool)

                # Apply re-masking: set selected positions back to mask_token_id
                x[to_mask] = mask_token_id

                # For positions that started as mask and were not re-masked, unmask them with sampled tokens
                keep_unmask = mask_index & (~to_mask)
                x[keep_unmask] = x0_full[keep_unmask]
            elif alg == "p2_upgraded":
                # p2_upgraded: only sample masked positions, but compute confidence for all positions
                # kappa_t: fraction of tokens to keep unmasked (can be replaced with custom schedule)
                kappa_t = (i + 1) / steps

                # Step 1: Sample ONLY masked positions
                # Use "p2" algorithm for sampling (same as "origin" - uses default confidence)
                confidence_masked, x0_masked = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="p2"
                )
                
                # Step 2: Update only masked positions with sampled tokens
                x[mask_index] = x0_masked

                # Step 3: Compute confidence for ALL positions
                # Use raw logits (no temp/top-p/top-k) so confidence reflects the
                # original model distribution, matching llada_sample behavior.
                probs_full = torch.softmax(logits.float(), dim=-1)
                
                # Get probability of current token at each position
                # For masked positions: x already contains x0_masked (just sampled)
                # For unmasked positions: x contains existing tokens
                prob_all = torch.gather(probs_full, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
                
                # Build confidence map: use prob_all for all variable positions
                # For "p2" algorithm, confidence is just the probability of the current token
                conf_full = prob_all.clone()
                
                # Mask out fixed positions (set to +inf so they're never selected for re-masking)
                full_conf = conf_full.clone()
                full_conf[fix_mask] = float("inf")
                # Prevent NaNs or extreme values from interfering
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )

                # Calculate how many positions to re-mask per sample
                # = number of variable positions * (1 - kappa_t)
                num_positions = (~fix_mask).sum(dim=1)  # [B]
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                # Boundaries: at least 0, at most total number of variable positions
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)

                # Select the lowest-confidence positions within (~fix_mask) for re-masking
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                max_k = int(num_to_mask.max().item())
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                    row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                    to_mask = torch.zeros_like(x, dtype=torch.bool)
                    batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                    valid_batch = batch_arange[row_mask]  # [sum_k]
                    valid_col   = topk_idx[row_mask]      # [sum_k]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x, dtype=torch.bool)

                # Apply re-masking: set selected positions back to mask_token_id
                x[to_mask] = mask_token_id
                
                # Note: Positions that were masked and not re-masked are already updated above
                # (x[mask_index] = x0_masked), so no additional update needed here
                
                
            elif alg == "p2_upgraded_ReMDM":
                # p2_upgraded_ReMDM: like p2_upgraded, but uses "unmask-time" confidence
                # Once a token is unmasked, its confidence is frozen until it's remasked
                kappa_t = (i + 1) / steps

                # Step 1: Sample ONLY masked positions (use temp/top-p/top-k for sampling)
                _, x0_masked = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="p2"
                )
                
                # Step 2: Update masked positions with sampled tokens
                x[mask_index] = x0_masked
                
                # Step 3: Compute unmask-time confidence using raw logits (matching p2_upgraded)
                # Use raw logits (no temp/top-p/top-k) so confidence reflects the
                # original model distribution, matching llada_sample behavior.
                probs_full = torch.softmax(logits.float(), dim=-1)
                
                # Get probability of current token at each position
                # For masked positions: x already contains x0_masked (just sampled)
                prob_all = torch.gather(probs_full, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
                
                # Record confidence for newly unmasked positions (unmask-time confidence)
                # These are the positions that were mask and are now unmasked
                confidence_history[mask_index] = prob_all[mask_index]

                # Step 4: Build confidence map for remasking decision
                # - For already unmasked positions: use confidence_history (frozen at unmask-time)
                # - For fixed positions: already set to inf in confidence_history
                
                # Start with confidence_history (contains unmask-time confidence for unmasked positions)
                full_conf = confidence_history.clone()
                
                # Note: After x[mask_index] = x0_masked, there should be no mask positions left
                # in x (for this step). But we still need to handle the remasking logic.
                # The positions that were just unmasked now have their confidence in confidence_history.
                
                # Prevent NaNs or extreme values from interfering
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )

                # Calculate how many positions to re-mask per sample
                num_positions = (~fix_mask).sum(dim=1)  # [B]
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)

                # Select the lowest-confidence positions within (~fix_mask) for re-masking
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                max_k = int(num_to_mask.max().item())
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                    row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                    to_mask = torch.zeros_like(x, dtype=torch.bool)
                    batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                    valid_batch = batch_arange[row_mask]  # [sum_k]
                    valid_col = topk_idx[row_mask]  # [sum_k]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x, dtype=torch.bool)

                # Apply re-masking: set selected positions back to mask_token_id
                x[to_mask] = mask_token_id
                
                # Step 6: Clear confidence_history for remasked positions
                # They are no longer "unmasked", so their unmask-time confidence is invalid
                confidence_history[to_mask] = float("-inf")

            elif alg in ["maskgit_plus", "entropy", "topk_margin"]:
                # Confidence-based sampling (maskgit, entropy, etc.)
                
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                confidence = confidence.to(mask_logits.dtype)

                # Calculate number of mask tokens per sample
                num_mask_tokens_per_sample = mask_index.sum(dim=1)  # [batch_size]
                
                # Calculate transfer tokens per sample
                if i < steps - 1:
                    number_transfer_tokens_per_sample = (num_mask_tokens_per_sample.float() * (1 - s / t)).long()
                else:
                    number_transfer_tokens_per_sample = num_mask_tokens_per_sample
                
                # Build full confidence matrix
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                # Get maximum transfer tokens for efficient batching
                max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
                
                if max_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        # Use topk for each sample
                        _, all_transfer_indices = torch.topk(full_confidence, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                    else:
                        # Robust vectorized sampling via Gumbel-TopK (no replacement)
                        # Handles rows with fewer valid positions than requested and rows with no valid positions
                        # full_confidence has -inf for invalid positions; keep them -inf so they won't be selected
                        scaled_logits = full_confidence / alg_temp
                        # Uniform in (0,1) to avoid log(0)
                        uniform = torch.rand_like(scaled_logits).clamp_(min=1e-20, max=1 - 1e-20)
                        gumbel_noise = -torch.log(-torch.log(uniform))
                        scores = scaled_logits + gumbel_noise
                        _, all_transfer_indices = torch.topk(scores, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                    
                    # Create mask for valid transfers (handle variable number of transfers per sample)
                    batch_size = x.size(0)
                    valid_mask = torch.arange(max_transfer_tokens, device=x.device).unsqueeze(0) < number_transfer_tokens_per_sample.unsqueeze(1)  # [batch_size, max_transfer_tokens]
                    
                    # Get valid transfer indices and corresponding batch indices
                    valid_transfer_indices = all_transfer_indices[valid_mask]  # [total_valid_transfers]
                    valid_batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(all_transfer_indices)[valid_mask]  # [total_valid_transfers]
                    
                    # Prepare the transfer data
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    
                    # Batch update using advanced indexing
                    x[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
            
            else:
                raise NotImplementedError(f"Algorithm {alg} not implemented.")

            if histories is not None:
                histories.append(x.clone())

        if generation_config.return_dict_in_generate:
            return MDMModelOutput(sequences=x, history=histories)
        else:
            return x