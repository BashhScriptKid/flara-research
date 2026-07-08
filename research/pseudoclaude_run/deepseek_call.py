#!/usr/bin/env python3
"""Stateless single-call helper for DeepSeek V4 Pro. Each invocation is a fresh
session with no memory of prior calls, by design (this is what the PseudoClaude
pipeline calls a 'fresh session' to prevent cross-phase contamination).

Usage:
    python3 deepseek_call.py --system system.txt --user user.txt --out out.json --max-tokens 8000
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


def call(system_text, user_text, max_tokens=8000, temperature=1.0, model="deepseek-v4-pro"):
    key = load_key()
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", help="path to system prompt text file")
    ap.add_argument("--user", required=True, help="path to user message text file")
    ap.add_argument("--out", required=True, help="path to write raw JSON response")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    system_text = ""
    if args.system:
        with open(args.system) as f:
            system_text = f.read()
    with open(args.user) as f:
        user_text = f.read()

    result = call(system_text, user_text, max_tokens=args.max_tokens, temperature=args.temperature, model=args.model)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    finish_reason = choice.get("finish_reason", "")
    usage = result.get("usage", {})

    content_path = args.out.replace(".json", "_content.md")
    with open(content_path, "w") as f:
        f.write(content)

    if reasoning:
        reasoning_path = args.out.replace(".json", "_reasoning.md")
        with open(reasoning_path, "w") as f:
            f.write(reasoning)

    print(f"finish_reason={finish_reason}", file=sys.stderr)
    print(f"usage={usage}", file=sys.stderr)
    print(f"content_chars={len(content)} reasoning_chars={len(reasoning)}", file=sys.stderr)
    if finish_reason == "length":
        print("WARNING: response truncated by max_tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
