#!/usr/bin/env python3
"""Convert game NPC dialogue CSV → JSONL for train_with_tokenizer.

CSV format:
    prompt,message,mood
    "สวัสดี","อืม... ว่ายังไงนะ","shy"
    ...

Output JSONL format (OpenAI chat format):
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python csv_to_jsonl.py dialogues.csv -o dialogues.jsonl
    python csv_to_jsonl.py dialogues.csv                  # same dir, auto name
"""

import argparse
import csv
import json
import os
import sys

SYSTEM_PROMPT = (
    "You are หมิว NPC. Always respond in JSON: "
    '{"message": "...", "mood": "..."}. '
    "moods: shy, happy, sad, angry, nervous, blush, calm"
)


def main():
    parser = argparse.ArgumentParser(description="Convert dialogue CSV → JSONL")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument(
        "-o", "--output", help="Output JSONL file (default: same dir, same name)"
    )
    parser.add_argument(
        "-n",
        "--num-repeats",
        type=int,
        default=1,
        help="Repeat each row N times with variations (default: 1)",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.csv_file)[0]
        out_path = base + ".jsonl"

    count = 0
    with (
        open(args.csv_file, "r", encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            msg = row.get("message", "").strip()
            mood = row.get("mood", "").strip()

            if not prompt or not msg or not mood:
                print(f"  skipping incomplete row: {row}", file=sys.stderr)
                continue

            assistant_content = json.dumps(
                {"message": msg, "mood": mood}, ensure_ascii=False
            )

            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ {count} samples → {out_path}")


if __name__ == "__main__":
    main()
