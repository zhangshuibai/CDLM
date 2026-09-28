"""
Strip a trailing markdown code fence from refined completions.

Why this is needed
------------------
Both fine-tuned arms are trained on Nemotron-SFT-Code, whose code is wrapped in
markdown ``` fences. The CRB `remove_all` setting leaves *every* body token free, and
the human-eval+ buggy bodies happen to end with an extra blank line (`...\n\n`) that
human-eval's do not. That spare trailing position gets filled with "```" by any model
that has seen fenced code:

    base  human-eval+  0.0%  of completions contain ```
    MDLM  human-eval+ 91.5%
    (human-eval: base 0.0%, MDLM 9.9%)

Under `--no_postprocess` that lone fence is a syntax error, so it zeroes the cell for
reasons that have nothing to do with error correction. This script removes the
artifact with one rule applied **identically to base, MDLM and CDLM**, so the
comparison stays controlled.

Rule: truncate the completion at the first occurrence of the substring "```",
then drop any trailing whitespace-only lines. Nothing else is touched.

(An earlier version truncated at the first *line* starting with "```". That missed
fences emitted mid-line, e.g. `    return results```:` , which is common at larger
edit budgets — it left CDLM's human-eval+ T=4 cells at 38.4 instead of recovering
them. The substring rule fixes that. Python code never legitimately contains "```".)
"""

import argparse
import json
import os


def defence(text):
    i = text.find("```")
    if i != -1:
        text = text[:i]
    return text.rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    n = changed = 0
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.in_file) as fi, open(args.out_file, "w") as fo:
        for line in fi:
            r = json.loads(line)
            n += 1
            for k in ("completion", "refined_completion"):
                if k in r and isinstance(r[k], str):
                    new = defence(r[k])
                    if new != r[k]:
                        if k == "completion":
                            changed += 1
                        r[k] = new
            fo.write(json.dumps(r) + "\n")
    print(f"defence: {changed}/{n} completions truncated -> {args.out_file}")


if __name__ == "__main__":
    main()
