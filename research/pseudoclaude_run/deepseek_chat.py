#!/usr/bin/env python3
"""Multi-turn conversation helper for DeepSeek V4 Pro. Maintains a JSON
message-list file across calls so a single 'session' (per the PseudoClaude
pipeline's terminology) can have multiple turns, while separate session
files remain fully isolated from each other (fresh-session requirement).

Usage:
    python3 deepseek_chat.py --conv conv.json --system system.txt --add-user user_turn.txt --max-tokens 16000
    (first call creates conv.json if absent and seeds it with --system)
"""
import argparse
import json
import os
import sys

import requests

API_URL = "https://api.deepseek.com/chat/completions"
KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deepseek_key")


def load_key():
    with open(KEY_PATH) as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True, help="path to conversation JSON file (created if absent)")
    ap.add_argument("--system", help="path to system prompt text file (only used if conv file doesn't exist yet)")
    ap.add_argument("--add-user", required=True, help="path to text file with the new user turn to append")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    if os.path.exists(args.conv):
        with open(args.conv) as f:
            messages = json.load(f)
    else:
        messages = []
        if args.system:
            with open(args.system) as f:
                messages.append({"role": "system", "content": f.read()})

    with open(args.add_user) as f:
        messages.append({"role": "user", "content": f.read()})

    key = load_key()
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    resp = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    result = resp.json()

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    finish_reason = choice.get("finish_reason", "")
    usage = result.get("usage", {})

    messages.append({"role": "assistant", "content": content})
    with open(args.conv, "w") as f:
        json.dump(messages, f, indent=2)

    turn_idx = len(messages) - 1
    out_base = args.conv.replace(".json", f"_turn{turn_idx}")
    with open(out_base + "_content.md", "w") as f:
        f.write(content)
    if reasoning:
        with open(out_base + "_reasoning.md", "w") as f:
            f.write(reasoning)

    print(f"turn={turn_idx} finish_reason={finish_reason}", file=sys.stderr)
    print(f"usage={usage}", file=sys.stderr)
    print(f"content_chars={len(content)} reasoning_chars={len(reasoning)}", file=sys.stderr)
    if finish_reason == "length":
        print("WARNING: response truncated by max_tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
