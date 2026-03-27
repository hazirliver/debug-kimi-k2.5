import json
import os
import sys

import httpx
from tqdm import tqdm

# --- Configuration ---
BASE_URL = "https://api.tokenfactory.me-west1.nebius.com/v1"
API_KEY = os.getenv("TF_API_KEY")
MODEL = "dedicated/opencode/Kimi-K2.5-Hivgg0CgJWFU"

if not API_KEY:
    print("ERROR: Set TF_API_KEY environment variable")
    sys.exit(1)

# --- Load session ---
with open("session.json") as f:
    session = json.load(f)

# --- Tool definitions (matching what the session uses) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer", "description": "Line offset"},
                    "limit": {"type": "integer", "description": "Number of lines"},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file on disk by replacing oldString with newString",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string"},
                    "newString": {"type": "string"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
]


def convert_session_to_openai_messages(session_data, stop_before_message_id=None):
    """
    Walk through session messages and convert to OpenAI chat format.
    Stops RIGHT BEFORE stop_before_message_id so we can replay that turn.

    Key: groups all tool calls within the same assistant message into a single
    assistant message with a `tool_calls` array, followed by individual tool
    result messages — as required by the OpenAI API.
    """
    openai_messages = [
        {
            "role": "system",
            "content": "You are a coding assistant. Use tools to read and edit files.",
        }
    ]

    for msg in session_data["messages"]:
        msg_id = msg["info"]["id"]
        role = msg["info"]["role"]

        # Stop before the message that produced the error
        if stop_before_message_id and msg_id == stop_before_message_id:
            break

        if role == "user":
            # Collect text + file content into one user message
            content_parts = []
            for part in msg["parts"]:
                if part["type"] == "text" and not part.get("synthetic"):
                    content_parts.append(part["text"])
                elif part["type"] == "file":
                    # Include file content — this is what the model saw
                    filename = part.get("filename", "unknown")
                    content_parts.append(f"[File: {filename}]")
                # Skip synthetic text (tool output echo) — it's redundant
                # with the tool result messages we construct below
            if content_parts:
                openai_messages.append(
                    {"role": "user", "content": "\n\n".join(content_parts)}
                )

        elif role == "assistant":
            # Collect ALL tool calls and text from this assistant message
            tool_calls_in_msg = []
            tool_results_in_msg = []
            text_parts = []

            for part in msg["parts"]:
                if part["type"] == "tool":
                    call_id = part["callID"]
                    tool_name = part["tool"]
                    state = part["state"]

                    if tool_name == "invalid":
                        # The model tried to call a tool but produced malformed JSON.
                        # The error text contains the original tool name in state.input.tool.
                        actual_tool = state.get("input", {}).get("tool", "read")
                        tool_calls_in_msg.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": actual_tool,
                                    "arguments": "{}",  # placeholder — original was malformed
                                },
                            }
                        )
                        tool_results_in_msg.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": state.get("output", "Tool call failed"),
                            }
                        )
                    else:
                        # Normal tool call
                        tool_input = state.get("input", {})
                        arguments = json.dumps(tool_input)
                        tool_calls_in_msg.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": arguments,
                                },
                            }
                        )
                        if state["status"] == "completed":
                            tool_output = state.get("output", "")
                        else:
                            tool_output = state.get("error", "Error occurred")
                        tool_results_in_msg.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": tool_output if tool_output else "",
                            }
                        )

                elif part["type"] == "text" and part.get("text", "").strip():
                    text_parts.append(part["text"].strip())

            # Emit: first the assistant message (with tool_calls if any),
            # then the tool result messages
            if tool_calls_in_msg:
                assistant_msg = {
                    "role": "assistant",
                    "content": " ".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls_in_msg,
                }
                openai_messages.append(assistant_msg)
                openai_messages.extend(tool_results_in_msg)
            elif text_parts:
                openai_messages.append(
                    {"role": "assistant", "content": " ".join(text_parts)}
                )

    return openai_messages


# --- Find ALL messages that produced invalid tool calls ---
error_message_ids = []
for msg in session["messages"]:
    for part in msg["parts"]:
        if part.get("type") == "tool" and part.get("tool") == "invalid":
            mid = msg["info"]["id"]
            if mid not in error_message_ids:
                error_message_ids.append(mid)

print(f"Found {len(error_message_ids)} messages with invalid tool calls:")
for mid in error_message_ids:
    print(f"  - {mid}")


# --- Streaming helper ---
def call_streaming(messages, tools, temperature=0.7):
    """
    Send a streaming chat completion request and assemble the response manually,
    exactly as a real client would. Returns (tool_calls, finish_reason, raw_chunks).

    tool_calls is a list of dicts:
        {"id": ..., "name": ..., "arguments": <raw string as assembled from stream>}
    """
    assembled_calls = {}  # index -> {"id", "name", "arguments"}
    finish_reason = None
    raw_lines = []

    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "stream": True,
        },
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            raw_lines.append(line)
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choice = chunk.get("choices", [{}])[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta", {})
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in assembled_calls:
                    assembled_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.get("id"):
                    assembled_calls[idx]["id"] += tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    assembled_calls[idx]["name"] += fn["name"]
                if fn.get("arguments"):
                    assembled_calls[idx]["arguments"] += fn["arguments"]

    tool_calls = list(assembled_calls.values()) if assembled_calls else []
    return tool_calls, finish_reason, raw_lines


# --- Run reproduction attempts for each error point ---
NUM_ATTEMPTS = 50

for error_idx, error_msg_id in enumerate(error_message_ids):
    print(f"\n{'=' * 60}")
    print(f"Error point {error_idx + 1}/{len(error_message_ids)}: {error_msg_id}")
    print(f"{'=' * 60}")

    messages = convert_session_to_openai_messages(
        session, stop_before_message_id=error_msg_id
    )

    # Debug: print what we built
    print(f"Built {len(messages)} messages for replay:")
    for i, m in enumerate(messages):
        role = m["role"]
        if m.get("tool_calls"):
            fns = [tc["function"]["name"] for tc in m["tool_calls"]]
            print(f"  [{i}] {role} -> tool_calls: {fns}")
        elif role == "tool":
            preview = m["content"][:80].replace("\n", "\\n")
            print(f"  [{i}] {role} -> {preview}...")
        else:
            preview = (m.get("content") or "")[:80].replace("\n", "\\n")
            print(f"  [{i}] {role} -> {preview}...")

    print(f"\nSending {NUM_ATTEMPTS} streaming attempts to model...")

    found = False
    for attempt in tqdm(
        range(1, NUM_ATTEMPTS + 1), desc=f"Error point {error_idx + 1}"
    ):
        try:
            tool_calls, finish_reason, _ = call_streaming(messages, tools)
        except Exception as e:
            print(f"  Attempt {attempt}: HTTP error - {e}")
            continue

        if not tool_calls:
            if attempt <= 3 or attempt % 10 == 0:
                print(f"  Attempt {attempt}: no tool calls (finish={finish_reason})")
            continue

        for tc in tool_calls:
            fn_name = tc["name"]
            raw_args = tc["arguments"]
            try:
                parsed = json.loads(raw_args)
                if attempt <= 3 or attempt % 10 == 0:
                    print(
                        f"  Attempt {attempt}: valid {fn_name}({json.dumps(parsed)[:100]})"
                    )
            except json.JSONDecodeError as e:
                print(f"\n{'!' * 60}")
                print(f"REPRODUCED on attempt {attempt}!")
                print(f"Function: {fn_name}")
                print(f"Raw arguments: {raw_args!r}")
                print(f"Parse error: {e}")
                print(f"Finish reason: {finish_reason}")
                print(f"{'!' * 60}\n")
                found = True
                break

        if found:
            break

    if not found:
        print(
            f"\nDid not reproduce for error point {error_idx + 1} in {NUM_ATTEMPTS} attempts."
        )

print(f"\n{'=' * 60}")
print("Done. Tips if still not reproduced:")
print("  - Increase NUM_ATTEMPTS")
print("  - Add max_tokens=50 to force truncation mid-argument")
print("  - Lower temperature to 0 to check deterministic path")
print(f"{'=' * 60}")
