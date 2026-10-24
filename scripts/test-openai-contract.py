#!/usr/bin/env python3
"""
Integration contract test using the official Python `openai` SDK.
Validates that Kestrel functions as a drop-in replacement for OpenAI endpoints.
"""

import argparse
import json
import sys
import urllib.request
from openai import OpenAI


def run_contract_test(port: int, model_name: str) -> None:
    base_url = f"http://127.0.0.1:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="kestrel-contract-key", timeout=10.0)

    print(f"[Python Contract] Connecting to {base_url} with model '{model_name}'...", flush=True)

    # 1. Test models listing
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f"[Python Contract] models.list() returned: {model_ids}", flush=True)
    assert model_name in model_ids, f"Model '{model_name}' not found in {model_ids}"

    # 2. Test non-streaming chat completion with session header
    session_id = "py-contract-session"
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello from official OpenAI python client!"}],
        extra_headers={"X-Kestrel-Session-Id": session_id},
    )
    assert completion.choices[0].message.role == "assistant"
    response_text = completion.choices[0].message.content or ""
    assert len(response_text) > 0, "Empty completion returned"
    print(f"[Python Contract] Non-streaming chat completion: '{response_text}'", flush=True)

    # 3. Test streaming chat completion with same session
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Stream response"}],
        stream=True,
        extra_headers={"X-Kestrel-Session-Id": session_id},
    )

    received_tokens = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            received_tokens.append(chunk.choices[0].delta.content)

    streamed_full = "".join(received_tokens)
    print(f"[Python Contract] Streaming tokens received: {received_tokens}", flush=True)
    assert len(streamed_full) > 0, "No content streamed from server"

    # 4. Query Kestrel session extension
    sessions_url = f"http://127.0.0.1:{port}/kestrel/sessions"
    with urllib.request.urlopen(sessions_url, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        ids = [s["id"] for s in data["sessions"]]
        assert session_id in ids, f"Session '{session_id}' not found in {ids}"
        print(f"[Python Contract] /kestrel/sessions confirmed persistence: {ids}", flush=True)

    # 5. Delete session via extension endpoint
    del_url = f"http://127.0.0.1:{port}/kestrel/sessions/{session_id}"
    req = urllib.request.Request(del_url, method="DELETE")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        del_data = json.loads(resp.read().decode())
        assert del_data["deleted"] is True
        print(f"[Python Contract] Successfully deleted session '{session_id}'", flush=True)

    print("[Python Contract] Contract test completed successfully!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kestrel OpenAI Contract Test")
    parser.add_argument("--port", type=int, required=True, help="Server port to connect to")
    parser.add_argument("--model", type=str, default="mock-qwen", help="Expected model identifier")
    args = parser.parse_args()

    run_contract_test(args.port, args.model)


if __name__ == "__main__":
    main()
