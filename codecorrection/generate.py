# Adding bugs to the code based on human_eval
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
import random
import argparse
import os
import keyword
import torch
import autopep8 # for fixing mbpp indent
import tokenize
import io
from typing import Tuple, List, Any
from tqdm import tqdm

def token_len(op: str, tokenizer) -> int:
    ids = tokenizer(op, add_special_tokens=False)["input_ids"]
    return len(ids)

def get_comment_ranges(code: str) -> List[Tuple[int, int]]:
    """
    Get character ranges of comments in code.
    Returns list of (start, end) character positions for comments.
    """
    comment_ranges = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                # tok.start and tok.end are (line, column) tuples
                # We need to convert to character positions
                lines = code.split('\n')
                start_char = sum(len(lines[i]) + 1 for i in range(tok.start[0] - 1)) + tok.start[1]
                end_char = sum(len(lines[i]) + 1 for i in range(tok.end[0] - 1)) + tok.end[1]
                comment_ranges.append((start_char, end_char))
    except (tokenize.TokenError, SyntaxError):
        # If code has syntax errors, return empty list (will allow all replacements)
        pass
    return comment_ranges

def is_in_comment(pos: int, comment_ranges: List[Tuple[int, int]]) -> bool:
    """Check if a character position is within any comment."""
    return any(start <= pos < end for start, end in comment_ranges)

def check_token_consistency(buggy_ids: List[int], tokenizer) -> bool:
    """
    Check if decoding and re-encoding the buggy IDs results in the same token sequence.
    This ensures token positions are preserved when the code is processed downstream.
    """
    try:
        buggy_code = tokenizer.decode(buggy_ids, skip_special_tokens=True)
        retokenized_ids = tokenizer(buggy_code, return_tensors="pt")["input_ids"][0].tolist()
        return buggy_ids == retokenized_ids
    except Exception:
        return False

def add_header(orig_data: str, prompt: str, canonical: str):
    """ Process the prompt for adding header"""
    if orig_data in ["human-eval", "human-eval+"]:
        trimmed = re.split(r'>>>', prompt, maxsplit=1)[0].rstrip()
        stripped = trimmed.strip()
        
        # Detect which quote style is used in the docstring
        double_quote_pos = stripped.find('"""')
        single_quote_pos = stripped.find("'''")
        
        # Determine the docstring quote type based on which appears first
        if double_quote_pos != -1 and (single_quote_pos == -1 or double_quote_pos < single_quote_pos):
            quote_style = '"""'
        elif single_quote_pos != -1:
            quote_style = "'''"
        else:
            # No docstring found, default to """
            quote_style = '"""'
        
        # Check if docstring is properly closed with the same quote style
        if not stripped.endswith(quote_style):
            trimmed = trimmed + f'\n    {quote_style}'
        return trimmed + "\n\n"
    elif orig_data in ["mbpp", "mbpp+"]:
        lines = canonical.splitlines()
        # find class
        class_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("class "):
                class_idx = i
                break

        def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def "))
        doc = re.split(r'>>>', prompt, maxsplit=1)[0].rstrip()
        stripped = doc.strip()
        if not stripped.startswith('"""'):
            doc = '"""' + '\n' + stripped
        if not stripped.endswith('"""'):
            doc = doc.rstrip() + '\n"""'
        indent = "    "
        doc = "\n".join(indent + line for line in doc.splitlines())

        if class_idx is not None:
            insert_pos = class_idx + 1
        else:
            insert_pos = def_idx + 1

        new_lines = lines[:insert_pos] + ["", doc, ""]
        return "\n".join(new_lines)
    else:
        raise ValueError(f"Error type {orig_data}")

def operator_bug(canonical: str, n_replace: int, tokenizer):
    """
    Inject operator substitution errors into canonical code.
    Automatically handles operators with any token count (1, 2, or more tokens).

    Args:
        canonical (str): Correct code
        n_replace (int): Number of operators to replace
        tokenizer: Tokenizer to use

    Returns:
        tuple: (buggy_code, error_positions, original_tokens) or (None, None, None)
    """
    n_replace = int(n_replace)
    
    # If n_replace is 0, return original code without any modifications
    if n_replace == 0:
        return canonical, [], []
    
    ops_cand = ["+", "-", "*", "/", "<", ">", ">=", "<=", "!=", "==", ">>", "<<"]
    
    # Group operators by token count
    ops_by_token_count = {}
    for op in ops_cand:
        token_count = len(tokenizer(op, add_special_tokens=False)["input_ids"])
        if token_count not in ops_by_token_count:
            ops_by_token_count[token_count] = []
        ops_by_token_count[token_count].append(op)
    
    if not ops_by_token_count:
        return None, None, None

    ids = tokenizer(canonical, return_tensors="pt")["input_ids"][0].clone().tolist()
    buggy_ids = ids[:]
    touched_indices = set()

    # Get comment ranges to skip operators in comments
    comment_ranges = get_comment_ranges(canonical)

    # Find all operator positions for all token counts
    op_positions = []
    for token_count, valid_ops in ops_by_token_count.items():
        for i in range(len(ids) - token_count + 1):
            token_raw = tokenizer.decode(ids[i:i+token_count], skip_special_tokens=True)
            token_core = token_raw.strip()
            
            if token_core in valid_ops:
                # Verify token count matches
                if len(tokenizer(token_core, add_special_tokens=False)["input_ids"]) == token_count:
                    # Check if operator is in comment by checking decoded text position
                    decoded_before = tokenizer.decode(ids[:i], skip_special_tokens=True)
                    op_char_start = len(decoded_before)
                    if is_in_comment(op_char_start, comment_ranges):
                        continue
                    
                    first_raw = tokenizer.decode(ids[i:i+1], skip_special_tokens=True)
                    last_raw = tokenizer.decode(ids[i+token_count-1:i+token_count], skip_special_tokens=True)
                    left_spaces = len(first_raw) - len(first_raw.lstrip())
                    right_spaces = len(last_raw) - len(last_raw.rstrip())
                    op_positions.append((i, token_core, token_count, left_spaces, right_spaces))
    
    if not op_positions:
        return None, None, None

    random.shuffle(op_positions)

    success = 0
    current_token_count = 0
    attempt = 0
    max_attempt = n_replace * 10
    error_positions = []
    original_tokens = []

    # Try to replace operators
    while current_token_count < n_replace and attempt < max_attempt:
        attempt += 1
        if not op_positions:
            break

        # Find a candidate that fits the remaining token budget
        remaining_budget = n_replace - current_token_count
        candidate_idx = -1
        
        for idx, (pos, core, token_count, left, right) in enumerate(op_positions):
            if token_count <= remaining_budget:
                # Check overlap
                if not any(p in touched_indices for p in range(pos, pos + token_count)):
                    candidate_idx = idx
                    break
        
        if candidate_idx == -1:
            # No suitable candidate found in current op_positions
            break
            
        # Pop the selected candidate
        pos, core, token_count, left, right = op_positions.pop(candidate_idx)

        # Find replacement candidates with same token count
        candidates = [
            op for op in ops_by_token_count.get(token_count, [])
            if op != core
        ]

        if not candidates:
            continue

        new_op = random.choice(candidates)
        new_text = (" " * left) + new_op + (" " * right)
        new_ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]

        # Replace tokens
        for k in range(token_count):
            buggy_ids[pos + k] = new_ids[k]
            touched_indices.add(pos + k)
        
        success += 1
        current_token_count += token_count
        error_positions.extend(range(pos, pos + token_count))
        original_tokens.append(core)
    
    if current_token_count != n_replace:
        return None, None, None
    
    # Strict check: number of affected tokens must match n_replace
    if len(error_positions) != n_replace:
        return None, None, None
    
    # Final consistency check
    if not check_token_consistency(buggy_ids, tokenizer):
        return None, None, None

    return tokenizer.decode(buggy_ids, skip_special_tokens=True), error_positions, original_tokens

def token_bug(canonical: str, n_replace: int, tokenizer: Any, error_type: str):
    """
    Performs token-level variable or numeric literal substitution.
    Ensures the token length of the replacement matches the target.
    """
    n_replace = int(n_replace)

    if n_replace == 0:
        return canonical, [], []
    # Initialization
    original_ids = tokenizer(canonical, return_tensors="pt")["input_ids"][0]
    original_len = len(original_ids)
    current_ids = original_ids.clone()
    code_str = canonical
    used_pairs = set() # Stores (target_text, replacement_text) pairs to avoid duplicates
    touched_indices = set() # Track modified token positions
    success = 0
    current_token_count = 0
    attempts = 0
    max_attempt = n_replace * 10
    error_positions = []
    original_elements = [] # Stores text of original tokens/literals

    if error_type == 'var':
        element_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
    elif error_type == 'literal':
        element_pattern = re.compile(r"\b\d+(\.\d+)?\b")
    else:
        raise ValueError("substitution_type must be 'variable' or 'literal'")

    # Get comment ranges to skip elements in comments
    comment_ranges = get_comment_ranges(code_str)

    # Pre-process all potential elements and group by token length
    length_groups = {} 
    all_elements_text = set([m.group(0) for m in element_pattern.finditer(code_str)])

    if error_type == 'var':
        all_elements_text = {v for v in all_elements_text if not keyword.iskeyword(v)}
        
    for text in all_elements_text:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        l_len = len(ids)
        if l_len not in length_groups:
            length_groups[l_len] = []
        length_groups[l_len].append((text, ids))

    # Iterative Substitution Loop
    while current_token_count < n_replace and attempts < max_attempt:
        attempts += 1
        
        matches = [m for m in element_pattern.finditer(code_str)]
        
        # Filter elements: for variables, remove keywords; skip comments.
        valid_matches = []
        for m in matches:
            text = m.group(0)
            is_valid = True
            if error_type == 'var' and keyword.iskeyword(text):
                is_valid = False
            # Skip if in comment
            if is_in_comment(m.start(), comment_ranges):
                is_valid = False
            
            if is_valid:
                valid_matches.append(m)
        
        remaining_budget = n_replace - current_token_count
        
        # Identify valid length groups that fit in remaining budget
        valid_lengths = [l for l in length_groups.keys() if l <= remaining_budget]
        
        # If fewer than 2 distinct elements remain (target + replacement) or no valid lengths
        if len(valid_lengths) == 0 or len(valid_matches) < 1:
            break

        # Prioritize random selection, but filtered by valid lengths
        # Filter valid_matches to only include those with valid lengths
        filtered_matches = []
        for m in valid_matches:
            # We need to check token length of this match
            # This is a bit expensive but necessary for strict budget
            m_text = m.group(0)
            m_ids = tokenizer(m_text, add_special_tokens=False)["input_ids"]
            if len(m_ids) in valid_lengths:
                filtered_matches.append(m)
        
        if not filtered_matches:
            # Try to pick a target from length_groups directly to be faster?
            # But we need position info from regex match.
            # If no matches fit budget, break
             break

        target_match = random.choice(filtered_matches)
        target_text = target_match.group(0)
        
        target_ids_isolated = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_len = len(target_ids_isolated)
        
        if target_len not in length_groups:
            continue
            
        # Filter candidates and avoid already used (target, replacement) pairs
        candidates = [
            (c_text, c_ids) for c_text, c_ids in length_groups[target_len] 
            if c_text != target_text and (target_text, c_text) not in used_pairs
        ]
        
        if not candidates:
            continue
            
        replacement_text, replacement_ids = random.choice(candidates)
        
        found_indices = []
        current_ids_list = current_ids.tolist()
        
        for i in range(len(current_ids_list) - target_len + 1):
            if current_ids_list[i : i+target_len] == target_ids_isolated:
                # Check if any position in this range is already touched
                if not any(p in touched_indices for p in range(i, i + target_len)):
                    found_indices.append(i)
                
        if not found_indices:
            continue

        replace_idx = random.choice(found_indices) 
        temp_ids = current_ids.clone()
        temp_ids[replace_idx : replace_idx+target_len] = torch.tensor(
            replacement_ids, dtype=temp_ids.dtype, device=temp_ids.device
        )
        
        temp_code = tokenizer.decode(temp_ids, skip_special_tokens=True)
        
        if temp_code == code_str:
            continue

        current_ids = temp_ids
        code_str = temp_code
        
        error_positions.extend(range(replace_idx, replace_idx + target_len))
        for k in range(target_len):
            touched_indices.add(replace_idx + k)
        
        individual_tokens = [tokenizer.decode([tid]) for tid in target_ids_isolated]
        original_elements.extend(individual_tokens)
        
        # Mark this (target, replacement) pair as used to avoid exact duplicates
        used_pairs.add((target_text, replacement_text))
        success += 1
        current_token_count += target_len

    if current_token_count != n_replace:
        return None, None, None

    # Final consistency check
    current_ids_list = current_ids.tolist()
    if not check_token_consistency(current_ids_list, tokenizer):
        return None, None, None

    return code_str, error_positions, original_elements

def remove_duplicates(items):
    """
    Remove duplicate items based on task_id + buggy_body content.
    
    Args:
        items: List of item dictionaries
    
    Returns:
        List of unique items (first occurrence kept)
    """
    seen = set()
    unique_items = []
    for item in items:
        task_id = item.get('task_id', '')
        buggy_body = item.get('buggy_body', '')
        key = (task_id, buggy_body)
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return unique_items

def fix_class_indent(flat_code):
    """
    For fixing class indentation
    """
    lines = flat_code.strip().split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('class '):
            indent_level = 0
        elif line.startswith('def '):
            indent_level = 1 if 'self' in line else 0
        
        if line.startswith(('elif ', 'else:', 'except:', 'finally:')):
            curr = max(0, indent_level - 1)
        else:
            curr = indent_level
            
        formatted_lines.append("    " * curr + line)

        if line.endswith(':'):
            indent_level += 1
        if line.startswith(('return', 'break', 'continue', 'raise', 'pass')):
            indent_level = max(0, indent_level - 1)

    return autopep8.fix_code("\n".join(formatted_lines), options={'aggressive': 1})

def load_samples_from_file(input_file, orig_data):
    """
    Load samples from input file (e.g., filtered passed samples).
    Converts file format to dataset-like format for processing.
    
    Args:
        input_file: Path to input jsonl file
        orig_data: Dataset name (human-eval, human-eval+, mbpp, mbpp+)
    
    Returns:
        List of samples in dataset-like format
    """
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            # Extract code from cleaned_code (remove prompt part)
            cleaned_code = item.get("cleaned_code", "")
            prompt = item.get("prompt", "")
            
            # For human-eval, cleaned_code = prompt + "\n" + code
            # Extract code part by removing prompt
            if orig_data in ["human-eval", "human-eval+"]:
                if cleaned_code.startswith(prompt):
                    # Preserve original indentation; only drop the prompt prefix
                    canonical = cleaned_code[len(prompt):]
                else:
                    # Fallback: use completion if cleaned_code doesn't match prompt
                    canonical = item.get("completion", "")
                    # Try to extract code from completion
                    if "def " in canonical:
                        # Find first function definition
                        def_idx = canonical.find("def ")
                        canonical = canonical[def_idx:]
            elif orig_data in ["mbpp", "mbpp+"]:
                # For mbpp, cleaned_code might be different format
                canonical = item.get("code", cleaned_code)
            else:
                canonical = cleaned_code
            
            # Create sample in dataset-like format
            sample = {
                "task_id": item["task_id"],
            }
            
            # Add dataset-specific fields
            if orig_data in ["human-eval", "human-eval+"]:
                sample.update({
                    "canonical_solution": canonical,
                    "prompt": prompt,
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                })
            elif orig_data == "mbpp":
                sample.update({
                    "code": canonical,
                    "text": item.get("text", prompt),
                    "test_list": item.get("test_list", []),
                    "test_setup_code": item.get("test_setup_code", ""),
                    "challenge_test_list": item.get("challenge_test_list", []),
                })
            elif orig_data == "mbpp+":
                sample.update({
                    "code": canonical,
                    "prompt": item.get("prompt", prompt),
                    "test_list": item.get("test_list", []),
                    "test": item.get("test", ""),
                    "test_imports": item.get("test_imports", ""),
                    "source_file": item.get("source_file", ""),
                })
            
            samples.append(sample)
    
    return samples


def save_data(ds: str, type: str, orig_data:str, out_path: str, data_num = None, n_replace = 1, tokenizer=None, deduplicate=False, input_file=None):
    """
    Main process for adding bugs and save data to .jsonl file.
    data_num controls how many buggy variants to generate per original sample.
    deduplicate: If True, remove duplicate items based on buggy_body content.
    input_file: If provided, load samples from file instead of dataset.
    """
    variants_per_sample = 1 if data_num is None else max(1, int(data_num))
    all_items = []

    # Hardcode exclusion list for datasets (tasks with buggy test cases or code)
    excluded_task_ids = []
    if orig_data == "human-eval+":
        excluded_task_ids = ["HumanEval/32"]
    elif orig_data in ["mbpp", "mbpp+"]:
        excluded_task_ids = ["342"]  # MBPP task 342 has structural bug in code

    # Load samples from file or dataset
    if input_file:
        print(f"Loading samples from file: {input_file}")
        samples = load_samples_from_file(input_file, orig_data)
        print(f"Loaded {len(samples)} samples from file")
    else:
        samples = ds["test"]

    total_samples = len(samples)
    print(f"Processing {total_samples} samples, generating {variants_per_sample} variants per sample...")
    
    for sample in tqdm(samples, desc="Generating buggy variants"):
        # Skip excluded tasks
        task_id_str = str(sample["task_id"])
        if task_id_str in excluded_task_ids:
            continue
        canonical = sample["canonical_solution"] if orig_data in ["human-eval", "human-eval+"] else sample["code"]
        # canonical = textwrap.dedent(canonical).rstrip("\n")

        # MBPP dataset uses "text" field, others use "prompt"
        prompt = sample["text"] if orig_data == "mbpp" else sample["prompt"]
        # get the body
        if orig_data in ["mbpp", "mbpp+"]:
            if "class " in canonical:
                canonical = fix_class_indent(canonical)
                header_part = add_header(orig_data, prompt, canonical)
                lines = canonical.splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith("    def "))
                body_lines = lines[start:]
                body_src = "\n".join(body_lines)
            else:
                canonical = autopep8.fix_code(canonical)
                header_part = add_header(orig_data, prompt, canonical)
                lines = canonical.splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith("def "))
                body_lines = lines[start+1:]
                body_src = "\n".join(body_lines)
        else:
            body_src = canonical
            header_part = add_header(orig_data, prompt, canonical)

        # body = textwrap.indent(textwrap.dedent(body_src).rstrip("\n"), "    ")
        body = body_src
        canonical = header_part + body if orig_data in ["human-eval", "human-eval+"] else canonical  # Adding header to the canonical answer if human-eval

        variants_generated = 0
        while variants_generated < variants_per_sample:
            # add bugs
            if type == "operator":
                buggy_part, error_positions, error_original_tokens = operator_bug(body, n_replace, tokenizer)
            elif type == "var":  # type == "var"
                buggy_part, error_positions, error_original_tokens = token_bug(body, n_replace, tokenizer, type)
            elif type == "literal":
                buggy_part, error_positions, error_original_tokens = token_bug(body, n_replace, tokenizer, type)
            else:
                raise ValueError(f"Unsupported error type: {type}")

            if buggy_part is None:
                break

            new_prompt = header_part + buggy_part

            # items
            if orig_data in ["human-eval", "human-eval+"]:
                item = {
                    "task_id": sample["task_id"],
                    "prompt": new_prompt,
                    "canonical_solution": canonical,
                    "test": sample["test"],
                    "entry_point": sample["entry_point"],
                    "error_positions": error_positions,
                    "error_original_tokens": error_original_tokens,
                    "function_head": header_part,
                    "buggy_body": buggy_part
                }
            elif orig_data == "mbpp":
                item = {
                    "task_id": sample["task_id"],
                    "text": new_prompt,
                    "code": canonical,
                    "test_list": sample["test_list"],
                    "test_setup_code": sample["test_setup_code"],
                    "challenge_test_list": sample["challenge_test_list"],
                    "error_positions": error_positions,
                    "error_original_tokens": error_original_tokens,
                    "function_head": header_part,
                    "buggy_body": buggy_part
                }
            elif orig_data == "mbpp+":
                item = {
                    "task_id": sample["task_id"],
                    "code": canonical,
                    "prompt": new_prompt,
                    "source_file": sample["source_file"],
                    "test_imports": sample["test_imports"],
                    "test_list": sample["test_list"],
                    "test": sample["test"],
                    "error_positions": error_positions,
                    "error_original_tokens": error_original_tokens,
                    "function_head": header_part,
                    "buggy_body": buggy_part
                }
            else:
                raise ValueError(f"Unsupported dataset: {orig_data}")

            all_items.append(item)
            variants_generated += 1

    # Remove duplicates if requested
    if deduplicate:
        original_count = len(all_items)
        all_items = remove_duplicates(all_items)
        removed_count = original_count - len(all_items)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate items (kept {len(all_items)} unique items)")

    # Write all items to file
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["human-eval", "human-eval+", "mbpp", "mbpp+"],
        help="Error original dataset, must be one of [human-eval, human-eval+, mbpp, mbpp+]"
    )
    parser.add_argument(
        "--error_type",
        type=str,
        required=True,
        choices=["operator", "var", "literal"],
        help="Error type: 'operator' (replaces any operator), 'var' (replaces variables)"
    )
    parser.add_argument(
        "--data_num", type=int,
        default=1,
        help="Number of buggy variants to generate per original sample"
    )
    parser.add_argument(
        "--n_replace", type=str,
        required=False,
        default=1,
        help="Replace n groups of unary/binary token"
    )
    parser.add_argument(
        "--data_path", type=str,
        required=False,
        default="buggy_datasets",
        help="Path for saving data (default: buggy_datasets)"
    )
    parser.add_argument(
        "--model_name", type=str,
        required=True,
        help="The model whose tokenizer will be used"
    )
    parser.add_argument(
        "--verbose", type=bool,
        required=False,
        default = False,
        help="Detailed description of error"
    )
    parser.add_argument(
        "--seed", type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="Remove duplicate items based on buggy_body content"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip generation if output file already exists"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Input jsonl file with passed samples (if provided, skip dataset loading)"
    )

    # TODO: Add position
    # Confidence based remasking, remask token length can be changed

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    model_name_clean = os.path.basename(args.model_name)
    variants_per_sample = 1 if args.data_num is None else max(1, int(args.data_num))
    out_path = os.path.join(
        args.data_path,
        f"{args.dataset}/{model_name_clean}_{args.error_type}_{variants_per_sample}_wrong_{args.n_replace}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Check if output file exists and skip if requested
    if args.skip_existing and os.path.exists(out_path):
        print(f"Output file {out_path} already exists, skipping generation (--skip_existing enabled)")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False
    )
    if "open-dcoder" in args.model_name.lower():
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # Load dataset only if input_file is not provided
    ds = None
    if not args.input_file:
        print(f"Loading {args.dataset} dataset...")
        if args.dataset == "human-eval":
            ds = load_dataset("openai_humaneval")
        elif args.dataset == "human-eval+":
            ds = load_dataset("evalplus/humanevalplus")
        elif args.dataset == "mbpp":
            ds = load_dataset("google-research-datasets/mbpp")
        elif args.dataset == "mbpp+":
            ds = load_dataset("evalplus/mbppplus")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    else:
        print(f"Using input file: {args.input_file}")
        # Create a dummy dataset object for compatibility
        class DummyDataset:
            def __init__(self):
                self.test = []
        ds = DummyDataset()
    
    save_data(ds, args.error_type, args.dataset, out_path, variants_per_sample, args.n_replace, tokenizer, args.deduplicate, args.input_file)


if __name__ == "__main__":
    main()