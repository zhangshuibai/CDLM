"""
Refine LLaDA on error code.
"""

import os
import sys
import json
import re
import torch
from datasets import load_dataset
from llada_sample import llada_sample
from utils import load_model_and_tokenizer, build_output_paths
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torch.distributed as dist

class RefineDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_sample_history(batch_history, sample_idx):
    """Extract history for a single sample from batch history."""
    fix_mask = batch_history['prompt']['fix_mask'][sample_idx]

    sample_history = {
        'prompt': {
            'fix_mask': fix_mask,
            'tokens': batch_history['prompt']['tokens'][sample_idx],
        },
        'steps': []
    }

    # Calculate offset for this sample in the flattened variable tensors
    batch_size = batch_history['prompt']['fix_mask'].shape[0]
    num_vars_per_sample = [
        (~batch_history['prompt']['fix_mask'][i]).sum().item()
        for i in range(batch_size)
    ]
    start_idx = sum(num_vars_per_sample[:sample_idx])
    end_idx = start_idx + num_vars_per_sample[sample_idx]

    # Extract data for each step
    for step in batch_history['steps']:
        remask_pos = step['remask_positions'][sample_idx]
        variable_mask = ~fix_mask
        conf_var = step['conf_variable'][start_idx:end_idx]
        prob_var = step['prob_variable'][start_idx:end_idx]
        sample_step = {
            'x0_variable': step['x0_variable'][start_idx:end_idx],
            'conf_variable': step['conf_variable'][start_idx:end_idx],
            'prob_variable': step.get('prob_variable')[start_idx:end_idx], 
            'remask_positions': remask_pos,
            'remask_variable_positions': remask_pos[variable_mask],
            'avg_error_conf': float(conf_var.mean().item()),
            'avg_error_prob': float(prob_var.mean().item()),
        }

        sample_history['steps'].append(sample_step)

    return sample_history


def parse_tokens_from_path(file_path):
    """Parse tokens value from file path like .../tokens128/..."""
    path_str = str(file_path)
    match = re.search(r'/tokens(\d+)/', path_str)
    if match:
        return int(match.group(1))
    # Fallback: try to extract from max_new_tokens if available
    return None


def load_input_samples(results_file):
    """Load failed samples from evaluated results."""
    failed_samples = []

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if item['test_passed']:
                continue

            failed_samples.append({
                'task_id': item.get('task_id', ''),
                'prompt': item.get('prompt', ''), 
                'code': item.get('code') or item.get('canonical_solution', ''),
                'error_message': item.get('error_message', ''),
                'error_positions': item.get('error_positions', ''),
                'function_head': item['function_head'],
                'buggy_body': item['buggy_body'],
                'error_original_tokens': item.get('error_original_tokens', []),
            })

    return failed_samples

def build_refine_prompt_remove_all(function_head):
    """
    Build refine prompt with everything removed except function_head.
    
    Only function_head is provided as fixed context, no instruction,
    no previous attempt, no error feedback.
    """
    # Only return function_head, no additional instruction
    refine_instruction = f"{function_head}\n"
    return refine_instruction


def extract_function_parts(prompt: str):
    """ Return function_head, buggy_body """
    def_idx = prompt.find("def ")
    if def_idx == -1:
        return "", prompt

    func_part = prompt[def_idx:]

    m = re.match(
        r"(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\):"
        r"(?:\n[ \t]+[ruRU]{0,2}['\"]{3}[\s\S]*?['\"]{3})?)",
        func_part
    )
    if m:
        head = m.group(1)
        body = func_part[len(head):]
        return head.rstrip() + "\n", body.lstrip()
    else:
        newline_idx = func_part.find("\n")
        if newline_idx == -1:
            return func_part, ""
        head = func_part[:newline_idx+1]
        body = func_part[newline_idx+1:]
        return head.rstrip() + "\n", body.lstrip()
    
def collate_fn_refine(
    batch, tokenizer, mask_id, pad_id, refine_setting='remove_all'
):
    """
    Collate function for refining with cleaned code initialization.
    
    Context structure:
        [prompt (context part)] + [code (completion part)]
        |<----- Fixed part (fix_mask=True) ----->|  |<-- Variable (False) -->|
    
    Args:
        batch: List of failed samples, where:
            'prompt' is the context (fixed part, including function head/sig)
            'code' is the failed generated code (variable part, the body)
            'error_message' is the error feedback
        tokenizer: Tokenizer
        mask_id: Mask token ID
        pad_id: Padding token ID
    """
    task_ids = [item['task_id'] for item in batch]

    # Only support remove_all and remove_all_without_initialization
    if refine_setting not in ('remove_all', 'remove_all_without_initialization'):
        raise ValueError(
            f"Invalid refine_setting: {refine_setting}. "
            "Only 'remove_all' and 'remove_all_without_initialization' "
            "are supported."
        )
    
    # Build refine prompts and tokenize
    refine_contexts = []
    buggy_bodies = []
    function_heads = []
    error_positions_list = []
    error_original_tokens_list = []
    refine_context_tokens_list = []
    buggy_body_tokens_list = []
    
    for item in batch:
        function_head = item['function_head']
        buggy_body = item['buggy_body']
        
        # Build refine prompt (only function_head)
        refine_prompt = build_refine_prompt_remove_all(function_head)
        refine_contexts.append(refine_prompt)
        buggy_bodies.append(buggy_body)
        function_heads.append(function_head)
        error_positions_list.append(item['error_positions'])
        error_original_tokens_list.append(item['error_original_tokens'])
        
        # Tokenize without special tokens to avoid BOS/EOS in the middle
        context_tokens = tokenizer(
            refine_prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        # For body tokens: use mask_id if remove_all_without_initialization,
        # otherwise use actual buggy_body tokens
        if refine_setting == 'remove_all_without_initialization':
            # First tokenize to get length, then replace with mask_id
            body_tokens_len = len(tokenizer(
                buggy_body, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0])
            body_tokens = torch.full(
                (body_tokens_len,), mask_id, dtype=torch.long
            )
        else:
            body_tokens = tokenizer(
                buggy_body, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        
        refine_context_tokens_list.append(context_tokens)
        buggy_body_tokens_list.append(body_tokens)
    
    # Collate sequences and masks
    input_ids_list = []
    fix_mask_list = []
    attention_mask_list = []
    
    for context_tokens, body_tokens in zip(
        refine_context_tokens_list, buggy_body_tokens_list
    ):
        full_seq = torch.cat([context_tokens, body_tokens])
        fix_mask = torch.cat([
            torch.ones(len(context_tokens), dtype=torch.bool),  # Fixed context
            torch.zeros(len(body_tokens), dtype=torch.bool),    # Variable body
        ])
        attn_mask = torch.ones(len(full_seq), dtype=torch.bool)
        
        input_ids_list.append(full_seq)
        fix_mask_list.append(fix_mask)
        attention_mask_list.append(attn_mask)
    
    max_len = max(seq.shape[0] for seq in input_ids_list)
    
    for i in range(len(input_ids_list)):
        current_len = input_ids_list[i].shape[0]
        if current_len < max_len:
            pad_len = max_len - current_len
            input_ids_list[i] = torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                input_ids_list[i]
            ])
            fix_mask_list[i] = torch.cat([
                torch.ones((pad_len,), dtype=torch.bool),
                fix_mask_list[i]
            ])
            attention_mask_list[i] = torch.cat([
                torch.zeros((pad_len,), dtype=torch.bool),
                attention_mask_list[i]
            ])
    
    return {
        'input_ids': torch.stack(input_ids_list).long(),
        'fix_mask': torch.stack(fix_mask_list),
        'attention_mask': torch.stack(attention_mask_list),
        'task_ids': task_ids,
        'refine_contexts': refine_contexts,
        'buggy_bodies': buggy_bodies,
        'function_heads': function_heads,
        'error_positions': error_positions_list,
        'error_original_tokens': error_original_tokens_list,
    }

def refine_main(
    initial_results_file,
    model_name,
    batch_size,
    refined_steps,
    algorithm,
    temperature,
    refine_setting,
    confidence_threshold=None,
    output_prefix="correction",
    mad_k=2.5,
    skip_existing=False
):
    """Refine failed HumanEval samples with error feedback."""
    # Load model and tokenizer
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    
    # Build output paths first to check if output file exists
    paths = build_output_paths(
        initial_results_file=initial_results_file,
        refined_steps=refined_steps,
        refine_setting=refine_setting,
        algorithm=algorithm,
        local_rank=local_rank,
        world_size=world_size,
        output_prefix=output_prefix,
        confidence_threshold=confidence_threshold,
        mad_k=mad_k,
        temperature=temperature
    )
    output_file = paths['output_file']
    history_dir = paths['history_dir']
    history_file = paths['history_file']
    
    # Check if output file exists and skip if requested
    if skip_existing and os.path.exists(output_file):
        print(
            f"[Rank {local_rank}] Output file {output_file} already exists, "
            f"skipping refinement (--skip_existing enabled)"
        )
        return
    
    model, tokenizer, pad_id, mask_id = load_model_and_tokenizer(model_name, device=device, local_rank=local_rank)
    
    # Load failed samples
    print(f"[Rank {local_rank}] Loading failed samples from {initial_results_file}...")
    failed_samples = load_input_samples(initial_results_file)
    print(f"[Rank {local_rank}] Found {len(failed_samples)} failed samples")
    
    if len(failed_samples) == 0:
        raise ValueError("No failed samples to refine!")
    
    # Distribute data across GPUs
    per_gpu = len(failed_samples) // world_size
    start_idx = local_rank * per_gpu
    if local_rank < world_size - 1:
        end_idx = start_idx + per_gpu
    else:
        end_idx = len(failed_samples)
    local_dataset = failed_samples[start_idx:end_idx]

    print(f"[Rank {local_rank}] Processing {len(local_dataset)} samples")

    # Create dataloader
    refine_dataset = RefineDataset(local_dataset)

    def collate_wrapper(batch):
        return collate_fn_refine(batch, tokenizer, mask_id, pad_id, refine_setting)

    dataloader = DataLoader(
        refine_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper
    )

    # Generate
    results = []
    all_histories = []

    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"[Rank {local_rank}] Refining"
    )):
        input_ids = batch['input_ids'].to(device).long()  # Ensure long type after moving to device
        fix_mask = batch['fix_mask'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate with history
        output = llada_sample(
            model=model,
            input_ids=input_ids,
            fix_mask=fix_mask,
            mask_id=mask_id,
            attention_mask=attention_mask,
            steps=refined_steps,
            algorithm=algorithm,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_history=True,
            model_name=model_name,
            mad_k=mad_k,
        )

        # Decode sequences
        sequences = output['sequences']
        generated_texts = tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )

        # Extract histories for each sample in batch
        batch_size_actual = len(batch['task_ids'])
        for i in range(batch_size_actual):
            sample_history = extract_sample_history(output['history'], i)
            # Add task_id to history for identification
            sample_history['task_id'] = batch['task_ids'][i]
            
            # Add error information for visualization
            error_positions = batch['error_positions'][i]
            error_original_tokens = batch['error_original_tokens'][i]
            sample_history['error_positions'] = error_positions
            sample_history['error_original_tokens'] = error_original_tokens

            # Calculate correct avg_error_conf/prob for each step (only for error positions)
            for step in sample_history['steps']:
                conf_var = step['conf_variable'].cpu().numpy()
                prob_var = step['prob_variable'].cpu().numpy()
                
                # Calculate average confidence/prob only for error positions
                valid_confs = [conf_var[pos] for pos in error_positions]
                valid_probs = [prob_var[pos] for pos in error_positions]
                
                step['avg_error_conf'] = float(sum(valid_confs) / len(valid_confs)) if len(valid_confs) > 0 else 0.0
                step['avg_error_prob'] = float(sum(valid_probs) / len(valid_probs)) if len(valid_probs) > 0 else 0.0

            all_histories.append(sample_history)

        # Save results (without history in jsonl)
        for i, (task_id, refine_ctx, buggy_body, function_head, text) in enumerate(zip(
            batch['task_ids'],
            batch['refine_contexts'],
            batch['buggy_bodies'],
            batch['function_heads'],
            generated_texts
        )):
            # Extract refined completion using fix_mask to stay aligned with buggy_body
            seq_tensor = sequences[i].detach().cpu()
            fix_mask_cpu = batch['fix_mask'][i].bool()
            attn_mask_cpu = batch['attention_mask'][i].bool()
            body_mask = (~fix_mask_cpu) & attn_mask_cpu
            body_token_ids = seq_tensor[body_mask]
            if body_token_ids.numel() == 0:
                refined_completion = ""
            else:
                refined_completion = tokenizer.decode(
                    body_token_ids.tolist(),
                    skip_special_tokens=True
                )

            result = {
                'task_id': task_id,
                'original_buggy_body': buggy_body,
                'refined_completion': refined_completion,
                'completion': function_head + refined_completion,  # Use 'completion' for evaluate_humaneval compatibility
                'refine_context': refine_ctx,
                'full_text': text,
                'actual_steps': output['actual_steps'][i].item(),
                'algorithm': algorithm,
                'steps': refined_steps,
            }
            results.append(result)

    # Save results (jsonl)
    output_path = f"{output_file}.rank{local_rank}"
    print(f"[Rank {local_rank}] Saving results to {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save histories (.pt)
    # History directory and file paths already defined above
    # (reuse the paths from the existence check)

    print(f"[Rank {local_rank}] Saving {len(all_histories)} histories to "
          f"{history_file}")
    torch.save(all_histories, history_file)

    print(f"[Rank {local_rank}] Done!")

    # Wait for all ranks (simple file-based synchronization)
    if world_size > 1:
        # Create sync file for this rank
        sync_file = f"{output_file}.sync.rank{local_rank}"
        with open(sync_file, 'w') as f:
            f.write('done')
        
        # Wait for all ranks to finish
        if local_rank == 0:
            import time
            print("Waiting for all ranks to finish...")
            while True:
                all_done = all(
                    os.path.exists(f"{output_file}.sync.rank{r}")
                    for r in range(world_size)
                )
                if all_done:
                    break
                time.sleep(1)
            print("All ranks finished!")

    # Merge results on rank 0
    if local_rank == 0:
        print("Merging results from all ranks...")
        all_results = []
        for rank in range(world_size):
            rank_file = f"{output_file}.rank{rank}"
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as f:
                    for line in f:
                        all_results.append(json.loads(line))
                os.remove(rank_file)

        # Save merged results
        with open(output_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        print(f"Saved {len(all_results)} refined results to {output_file}")
        print(f"Histories saved to {history_dir}/")
        
        # Clean up sync files
        for rank in range(world_size):
            sync_file = f"{output_file}.sync.rank{rank}"
            if os.path.exists(sync_file):
                os.remove(sync_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_results_file", type=str,
        default="eval_results/humaneval_evaluated.jsonl",
        help="Path to evaluated results with failed samples"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="GSAI-ML/LLaDA-8B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--refined_steps", type=int, default=2,
        help="Number of refinement steps"
    )
    parser.add_argument(
        "--algorithm", type=str,
        default='self_conf-remask:vanilla'
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--refine_setting", type=str, default="remove_all",
        choices=["remove_all", "remove_all_without_initialization"],
        help=(
            "Refine setting: 'remove_all' (only function_head, "
            "body initialized with original code), or "
            "'remove_all_without_initialization' (only function_head, "
            "body initialized with mask_id)"
        )
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Optional confidence threshold (e.g., 0.9) for remasking strategy"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="correction",
        help="Prefix for output directories (default: 'correction'). Use 'AR_correction' for AR experiments."
    )
    parser.add_argument(
        "--mad_k",
        type=float,
        default=2.5,
        help="Threshold multiplier for MAD algorithm (default: 2.5, typical values: 2.0, 2.5, 3.0). Only used when algorithm is 'self_conf-remask:vanilla_MAD'"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip refinement if output file already exists"
    )

    args = parser.parse_args()

    refine_main(
        initial_results_file=args.initial_results_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        refined_steps=args.refined_steps,
        algorithm=args.algorithm,
        temperature=args.temperature,
        refine_setting=args.refine_setting,
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
        mad_k=args.mad_k,
        skip_existing=args.skip_existing
    )

