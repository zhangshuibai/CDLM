"""
Evaluate HumanEval and MBPP results using HuggingFace code_eval and referred EvalPlus.
"""

import json
import argparse
import os
from pathlib import Path
import subprocess
from datetime import datetime
from tqdm import tqdm

# Enable code evaluation (must be set before importing evaluate)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import evaluate as hf_evaluate  # noqa: E402
from sanitize import sanitize  # noqa: E402


def load_results(results_file):
    """Load results from jsonl file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def write_jsonl(data, file_path):
    """Write data to jsonl file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def print_statistics(results, use_prompt=False):
    """Print statistics about generation."""
    print(f"\nTotal samples: {len(results)}")

    if 'actual_steps' in results[0]:
        steps_list = [r['actual_steps'] for r in results]
        print(f"Steps: min={min(steps_list)}, max={max(steps_list)}, "
              f"avg={sum(steps_list)/len(steps_list):.1f}")

    # Use prompt or completion based on flag
    field_name = 'prompt' if use_prompt else 'completion'
    if field_name in results[0]:
        code_lens = [len(r[field_name]) for r in results]
        print(f"Code length (chars): min={min(code_lens)}, "
              f"max={max(code_lens)}, "
              f"avg={sum(code_lens)/len(code_lens):.1f}")

        # Check if any codes are empty
        empty = sum(1 for c in code_lens if c == 0)
        if empty > 0:
            print(f"Warning: {empty} empty {field_name}s")

def pass_at_1(references, predictions, pass_at_k_metric, num_workers=100, timeout=3.0):
    """Compute pass@1 for a single sample.

    Args:
        references: Test cases
        predictions: Generated code
        pass_at_k_metric: HuggingFace code_eval metric
        num_workers: Number of workers for parallel evaluation (default: 100)
        timeout: Timeout per test in seconds (default: 3.0)

    Returns:
        tuple: (pass@1_score, detailed_feedback)
            - pass@1_score: float, 0.0 or 1.0
            - detailed_feedback: dict with test execution details including
              error messages
    """
    result = pass_at_k_metric.compute(
        references=references,
        predictions=predictions,
        k=[1],
        num_workers=num_workers,
        timeout=timeout,
    )
    # Result is a tuple: (metrics, detailed_results)
    metrics = result[0]
    detailed = result[1]

    # Extract feedback information with error details
    feedback = {'passed': False, 'result': 'unknown', 'error': None}
    if detailed and 0 in detailed:
        test_result = detailed[0][0][1]  # Get first test result

        # Copy all fields from test_result to feedback
        feedback = dict(test_result)

        # Add a human-readable error summary for convenience
        passed = test_result['passed']
        result_str = test_result['result']

        if not passed:
            # Extract stderr and stdout first as they contain most details
            stderr = test_result.get('stderr', '').strip()
            stdout = test_result.get('stdout', '').strip()

            error_msg = None
            if result_str.startswith('failed:'):
                # Remove "failed: " prefix
                error_msg = result_str[8:].strip()

            # Build comprehensive error message with all available info
            error_parts = []

            # Add the basic error message if exists
            if error_msg:
                error_parts.append(f"Error: {error_msg}")

            # Add stderr (usually has traceback/exception info)
            if stderr:
                error_parts.append(f"Stderr:\n{stderr}")

            # Add stdout (might have print statements or assertion messages)
            if stdout:
                error_parts.append(f"Stdout:\n{stdout}")

            # Combine all parts with clear separators
            if error_parts:
                feedback['error'] = "\n---\n".join(error_parts)
            else:
                # Last resort: show the full test_result for debugging
                feedback['error'] = (
                    f"Test failed but no detailed error info available.\n"
                    f"Result: {result_str}"
                )

    return metrics["pass@1"], feedback


def extract_code_from_completion(completion):
    """Extract code from completion (handle markdown code blocks)."""
    # Check if completion contains markdown code block
    if '```python' in completion:
        # Extract code from markdown block
        code = completion.split('```python\n', 1)[-1].split('```')[0]
    elif '```' in completion:
        # Generic code block
        code = completion.split('```', 1)[-1].split('```')[0]
    else:
        # No markdown, use as is
        code = completion
    # Only strip trailing whitespace to preserve indentation
    return code.rstrip()


def simulate_stop(code, stop_tokens=None):
    """
    Filter out content after certain stop tokens (e.g., '\ndef').

    Args:
        code: The code string to process
        stop_tokens: List of tokens to stop at.
            Default includes: '\ndef', '\nclass', '\nif __name__',
            '\n# language:', '\n# Language:'

    Returns:
        Code string truncated at the first occurrence of any stop token
    """
    if stop_tokens is None:
        stop_tokens = [
            '\ndef',
            '\nclass',
            '\nif __name__',
            '\n# language:',
        ]

    # Find the earliest position of any stop token
    positions = []
    code_lower = code.lower()

    for stop_token in stop_tokens:
        # Case-insensitive search for language comments
        if '# language:' in stop_token.lower():
            pos = code_lower.find(stop_token.lower())
        else:
            pos = code.find(stop_token)

        if pos != -1:
            positions.append(pos)

    # If any stop token found, truncate at the earliest position
    if positions:
        min_pos = min(positions)
        return code[:min_pos].rstrip()

    # No stop token found, return original code
    return code

def extract_relevant_prompt(prompt):
    """
    Keep docstring + imports + first function block
    """
    lines = prompt.strip().split('\n')
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            return line.rstrip()
    return lines[0] if lines else ""

def evaluate_main(
    results_file, output_file=None, use_sanitize=True,
    use_simulate_stop=False, dataset="human-eval",
    no_postprocess=False, n_workers=100,
    map_prompt2completion=False, timeout=3.0, skip_if_exist=False,
    summary_file=None, summary_metadata=None
):
    """
    Evaluate HumanEval and MBPP results based on HuggingFace code_eval.

    Args:
        results_file: Path to results jsonl file
        output_file: Path to save detailed results
            (optional, auto-generated if None)
        use_sanitize: Whether to use sanitize function to clean code
        use_simulate_stop: Whether to filter out content after stop tokens
            (e.g., '\ndef')
        no_postprocess: If True, skip all postprocessing and use completion as-is
            (useful for ground truth evaluation)
        n_workers: Number of workers for parallel evaluation (default: 100)
        skip_if_exist: If True, skip evaluation if output_file already exists (default: False)
    """
    # Auto-generate output_file if not provided
    if output_file is None:
        results_path = Path(results_file)
        # Add _evaluated suffix before .jsonl extension
        output_file = str(
            results_path.parent / f"{results_path.stem}_evaluated.jsonl"
        )
        print(f"Auto-generated output path: {output_file}")
    
    # Check if output_file exists and skip if requested
    if skip_if_exist and os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        print("Skipping evaluation (--skip_if_exist is enabled)")
        return None, []
    
    print("\n" + "=" * 60)
    print(f"Loading local dataset for {dataset} ...")
    print("=" * 60)

    task_id_to_info = {}

    # Load from HuggingFace datasets
    from datasets import load_dataset as hf_load_dataset
    
    if dataset == "human-eval":
        ds = hf_load_dataset("openai_humaneval")
    elif dataset == "human-eval+":
        ds = hf_load_dataset("evalplus/humanevalplus")
    elif dataset == "mbpp":
        ds = hf_load_dataset("google-research-datasets/mbpp")
    elif dataset == "mbpp+":
        ds = hf_load_dataset("evalplus/mbppplus")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    print(f"Loaded {dataset} from HuggingFace datasets")
    
    for sample in ds["test"]:
        tid = str(sample["task_id"])
        task_id_to_info[tid] = sample
    
    # Process task_id_to_info to extract only needed fields
    processed_info = {}
    for tid, sample in task_id_to_info.items():
        # MBPP uses "text" field, others use "prompt"
        if dataset == "mbpp":
            prompt_value = sample["text"]
        else:
            prompt_value = sample["prompt"]
        
        if dataset in ["human-eval", "human-eval+"]:
            processed_info[tid] = {
                "test": sample["test"],
                "entry_point": sample["entry_point"],
                "prompt": prompt_value,
            }
        elif dataset == "mbpp":
            processed_info[tid] = {
                "test_list": sample["test_list"],
                "test_setup_code": sample["test_setup_code"],
                "prompt": prompt_value,
                "entry_point": None,
            }
        elif dataset == "mbpp+":
            processed_info[tid] = {
                "prompt": prompt_value,
                "entry_point": None,
                "test_list": sample["test_list"],
                "test_setup_code": sample["test_imports"],
            }
    
    task_id_to_info = processed_info

    print(f"Loaded {len(task_id_to_info)} test samples")
    # Load results
    print(f"\nLoading results from {results_file}...")
    results = load_results(results_file)
    print_statistics(results, use_prompt=map_prompt2completion)

    # Load code_eval metric
    print(f"\n{'='*60}")
    print("Loading code_eval metric...")
    print(f"{'='*60}")
    pass_at_k_metric = hf_evaluate.load("code_eval")

    # Evaluate each sample
    print(f"\n{'='*60}")
    print("Evaluating samples...")
    print(f"{'='*60}")

    detailed_results = []
    pass_at_1_scores = []

    for result in tqdm(results, desc="Evaluating", unit="sample"):
        task_id = str(result['task_id']).strip()
        # Use prompt if map_prompt2completion is enabled, otherwise use completion
        if map_prompt2completion:
            # MBPP uses "text" field, others use "prompt"
            if dataset == "mbpp":
                completion = result["text"]
            else:
                completion = result["prompt"]
        else:
            completion = result["completion"]
        if task_id not in task_id_to_info:
            print(f"Warning: task_id {task_id} not found in dataset")
            raise ValueError(f"Task ID {task_id} not found in dataset")

        info = task_id_to_info[task_id]
        entry_point = info["entry_point"]
        prompt = info['prompt']

        # If no_postprocess is enabled, use completion as-is without any processing
        if no_postprocess:
            cleaned_code = completion
        else:
            # # Extract code from completion
            # code = extract_code_from_completion(completion)

            # # Apply simulate_stop if enabled (filter out content after stop tokens)
            # if use_simulate_stop:
            #     code = simulate_stop(code)

            # # Filter the prompt to include only the relevant part starting from 'def'
            # filtered_prompt = extract_relevant_prompt(prompt)

            # # If completion already contains the entry point definition (e.g. canonical solution),
            # # use it as-is to preserve import order; otherwise prepend filtered prompt.
            # entry_def_pattern = f"def {entry_point}"
            # if entry_point and entry_def_pattern in code:
            #     cleaned_code = code
            # else:
            #     cleaned_code = filtered_prompt + "\n" + code

            # # Sanitize cleaned_code to remove duplicate function definitions
            # # sanitize uses AST to parse and deduplicate definitions,
            # # keeping only the last definition for each function name
            # if use_sanitize and entry_point:
            #     try:
            #         sanitized = sanitize(cleaned_code, entrypoint=entry_point)
            #         # Only use sanitized result if it's not empty
            #         # sanitize may return empty string if all definitions
            #         # are filtered out
            #         if sanitized.strip():
            #             cleaned_code = sanitized
            #     except (SyntaxError, ValueError, Exception):
            #         # If sanitization fails (e.g., syntax error), use original code
            #         # This ensures evaluation can still proceed
            #         pass
            raise NotImplementedError("no_postprocess is not implemented")


        # Prepare for evaluation
        # references should be test code that calls check()
        # with the entry_point
        if dataset in ["human-eval", "human-eval+"]:
            test_cases = info["test"]
            test_code_with_call = test_cases + f"\ncheck({entry_point})"
        elif dataset in ["mbpp", "mbpp+"]:
            test_setup = info["test_setup_code"]
            test_list = info["test_list"]
            
            parts = []
            
            # Helper function to recursively flatten the data
            def flatten_data(item):
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, list):
                    for sub_item in item:
                        flatten_data(sub_item)
            # 1. Flatten test_setup_code
            if test_setup:
                flatten_data(test_setup)

            # 2. Flatten test_list
            flatten_data(test_list)
            
            # The error occurred here, but now 'parts' is guaranteed to contain only strings
            test_code_with_call = "\n".join(parts)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        references = [test_code_with_call]
        predictions = [[cleaned_code]]

        # Compute pass@1 and get feedback
        score, feedback = pass_at_1(references, predictions, pass_at_k_metric, n_workers, timeout)
        pass_at_1_scores.append(score)

        # Save detailed result with feedback including error messages
        detailed_result = {
            'task_id': task_id,
            'completion': completion,
            'cleaned_code': cleaned_code,
            'pass@1': score,
            'test_passed': feedback['passed'],
            'test_result': feedback['result'],
        }

        # Add comprehensive error message with all feedback info if test failed
        if not feedback['passed']:
            # Build comprehensive error message with all feedback
            error_parts = []

            # Add all feedback fields
            for key, value in feedback.items():
                if value and key != 'passed':  # Skip empty values & 'passed'
                    if isinstance(value, str) and value.strip():
                        error_parts.append(f"{key}: {value}")
                    elif not isinstance(value, str):
                        error_parts.append(f"{key}: {value}")

            if error_parts:
                error_msg = "\n---\n".join(error_parts)
                detailed_result['error_message'] = error_msg
            else:
                detailed_result['error_message'] = "Test failed: no details"

        # Also save individual stderr and stdout for easy access
        if 'stderr' in feedback and feedback['stderr']:
            detailed_result['stderr'] = feedback['stderr']
        if 'stdout' in feedback and feedback['stdout']:
            detailed_result['stdout'] = feedback['stdout']

        # Include additional info if available
        if 'actual_steps' in result:
            detailed_result['actual_steps'] = result['actual_steps']
        if 'algorithm' in result:
            detailed_result['algorithm'] = result['algorithm']
        
        # Pass through function_head and buggy_body from the original result file (from generate.py)
        if 'function_head' in result:
            detailed_result['function_head'] = result['function_head']
        if 'buggy_body' in result:
            detailed_result['buggy_body'] = result['buggy_body']
        
        # Pass through error_positions and error_original_tokens if available
        if 'error_positions' in result:
            detailed_result['error_positions'] = result['error_positions']
        if 'error_original_tokens' in result:
            detailed_result['error_original_tokens'] = result['error_original_tokens']
        
        # Also save prompt for backward compatibility
        detailed_result['prompt'] = prompt

        detailed_results.append(detailed_result)

    # Compute final metrics
    final_pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(pass_at_1_scores)}")
    print(f"Pass@1: {final_pass_at_1:.4f} ({final_pass_at_1*100:.2f}%)")
    print(f"Passed: {sum(pass_at_1_scores)} / {len(pass_at_1_scores)}")

    # Save detailed results (output_file is always set now)
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(detailed_results, output_file)
    print(f"\nDetailed results saved to {output_file}")

    # Save summary if requested
    if summary_file:
        save_summary(
            summary_file, final_pass_at_1, len(pass_at_1_scores),
            sum(pass_at_1_scores), summary_metadata, results_file
        )

    return final_pass_at_1, detailed_results


def save_summary(summary_file, pass_at_1, total_count, passed_count, metadata, results_file):
    """Save pass@1 summary to JSON file."""
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count lines in results_file
    results_file_line_count = 0
    if results_file and Path(results_file).exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_file_line_count = sum(1 for _ in f)
        except Exception as e:
            raise e
    
    # Parse metadata
    if metadata:
        # Parse metadata (format: "dataset:human-eval,error_type:operator,n_replace:1,model_name:...,data_num:2,...")
        metadata_dict = {}
        for item in metadata.split(','):
            if ':' in item:
                key, value = item.split(':', 1)
                metadata_dict[key.strip()] = value.strip()
    else:
        metadata_dict = {}
    
    # Prepare summary entry
    summary_entry = {
        'dataset': metadata_dict.get('dataset', ''),
        'error_type': metadata_dict.get('error_type', ''),
        'n_replace': metadata_dict.get('n_replace', ''),
        'pass_at_1': pass_at_1,
        'passed_count': passed_count,
        'total_count': total_count,
        'model_name': metadata_dict.get('model_name', ''),
        'data_num': metadata_dict.get('data_num', ''),
        'refined_steps': metadata_dict.get('refined_steps', ''),
        'algorithm': metadata_dict.get('algorithm', ''),
        'confidence_threshold': metadata_dict.get('confidence_threshold', ''),
        'temperature': metadata_dict.get('temperature', ''),
        'refine_setting': metadata_dict.get('refine_setting', ''),
        'results_file_line_count': results_file_line_count,
        'timestamp': datetime.now().isoformat(),
        'results_file': results_file
    }
    
    # Load existing entries if file exists
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_data = []
    else:
        existing_data = []
    
    # Append new entry
    existing_data.append(summary_entry)
    
    # Write back to file
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file", type=str,
        default="eval_results/humaneval_results.jsonl",
        help="Path to results jsonl file"
    )
    parser.add_argument(
        "--output_file", type=str,
        default=None,
        help=(
            "Path to save detailed evaluation results "
            "(auto-generated if not provided)"
        )
    )
    parser.add_argument(
        "--no_sanitize", action="store_true",
        help="Disable code sanitization"
    )
    parser.add_argument(
        "--use_simulate_stop", action="store_true",
        help=(
            "Filter out content after stop tokens "
            "(e.g., '\\ndef', '\\nclass')"
        )
    )
    parser.add_argument(
        "--n_workers", type=int, default=100,
        help="Workers for official evaluation (default: 100)"
    )
    parser.add_argument(
        "--timeout", type=float, default=8,
        help="Timeout per test for official evaluation (default: 8.0 seconds)"
    )
    parser.add_argument(
        "--dataset", 
        choices=["human-eval","human-eval+","mbpp","mbpp+"],
        default="human-eval",
        help="Timeout per test for official evaluation"
    )
    parser.add_argument(
        "--no_postprocess", action="store_true",
        help=(
            "Skip all postprocessing (extraction, sanitization, etc.) "
            "and use completion as-is. Useful for ground truth evaluation."
        )
    )
    parser.add_argument(
        "--map_prompt2completion", action="store_true",
        help=(
            "Use 'prompt' field instead of 'completion' field from results. "
            "Useful when results file uses 'prompt' to store the code (e.g., from generate.py)."
        )
    )
    parser.add_argument(
        "--skip_if_exist", action="store_true",
        help="Skip evaluation if output_file already exists (default: False)"
    )
    parser.add_argument(
        "--summary_file", type=str, default=None,
        help="Path to CSV file to append pass@1 summary (optional)"
    )
    parser.add_argument(
        "--summary_metadata", type=str, default=None,
        help="Metadata for summary in format 'dataset:xxx,error_type:xxx,n_replace:xxx'"
    )

    args = parser.parse_args()

    evaluate_main(
        results_file=args.results_file,
        output_file=args.output_file,
        use_sanitize=not args.no_sanitize,
        use_simulate_stop=args.use_simulate_stop,
        dataset=args.dataset,
        no_postprocess=args.no_postprocess,
        n_workers=args.n_workers,
        map_prompt2completion=args.map_prompt2completion,
        timeout=args.timeout,
        skip_if_exist=args.skip_if_exist,
        summary_file=args.summary_file,
        summary_metadata=args.summary_metadata
    )
