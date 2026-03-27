# Kimi-K2.5 Streaming Tool Call Truncation Bug

Reproduction script for a server-side streaming bug in the Kimi-K2.5 inference
endpoint where tool call arguments are silently truncated mid-JSON over SSE.

## The Bug

When using the streaming API (`stream: true`), the server occasionally sends
`[DONE]` before the closing `}` of a tool call's `arguments` JSON object.
The `finish_reason` is reported as `tool_calls` (normal completion) despite
the output being truncated — making the bug silent and hard to detect on the
server side.

**Observed truncated outputs:**
```
{"filePath": "...", "offset": 225, "limit":        ← value missing, no }
{"filePath": "...", "offset": 225, "limit": 35.    ← period instead of }, no }
{"filePath": "...", "offset": 220, "limit": 50     ← no closing }
```

The bug becomes **more frequent and eventually deterministic as context grows**:

| Context length | Reproduction rate |
|---|---|
| 8 messages  | ~3%  (1 in 30) |
| 15 messages | ~20% (1 in 5)  |
| 17 messages | 100% (always)  |

**Key finding:** The bug does NOT reproduce with non-streaming requests.
It is specific to SSE stream assembly.

## Setup

**Requirements:** Python 3.14+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

**API key:**
```bash
export TF_API_KEY="your_key_here"
```

## Reproducing

```bash
uv run python debug_opencode.py
```

The script will:
1. Parse `session.json` to find all 3 error points (messages containing `"tool": "invalid"`)
2. Rebuild the conversation history up to each error point
3. Send 50 streaming requests per error point
4. Print `REPRODUCED` with the raw truncated arguments when the bug triggers

**Expected output for error point 3** (reproduces on attempt 1, always):
```
REPRODUCED on attempt 1!
Function: read
Raw arguments: '{"filePath": "/Users/frank/.../lookup-user.ts", "offset": 225, "limit": '
Parse error: Expecting value: line 1 column 113 (char 112)
Finish reason: tool_calls
```

## Session

`session.json` is a real opencode session (`ses_2d8038df2ffe6EGnIVMW81Juv9`) that
triggered the bug three times while editing a TypeScript file. The three error
messages are:

| Message ID | Tool call | Error |
|---|---|---|
| `msg_d27fd8672001wTfT3hRzLErc1N` | `read` | `"limit": 35.` — missing `}` |
| `msg_d27ff3eb6001iGRwyKEWIDMul5` | `read` | `"limit": .` — truncated mid-value |
| `msg_d27ff449a001OWuVIZgXGrq7Nf` | `read` | `"limit": ` — value never emitted |

## How the script works

`debug_opencode.py` converts the opencode session format into OpenAI-compatible
messages, then replays each turn via the streaming API:

- **Message conversion:** Groups all tool calls within one assistant turn into a
  single `tool_calls` array (required by the OpenAI API), followed by individual
  `tool` result messages.
- **Streaming assembly:** Accumulates `delta.tool_calls[].function.arguments`
  chunks exactly as a real client would. If the assembled string fails
  `json.loads()`, the bug is confirmed.
- **Invalid tool handling:** For `"tool": "invalid"` parts in the session, the
  original tool name is extracted from `state.input.tool` and the error message
  is forwarded as the tool result so the model sees the same context it saw
  originally.

## Logs

See [`logs/run_2026-03-27.txt`](logs/run_2026-03-27.txt) for the full run output
including both the non-streaming baseline (0/150 reproductions) and the streaming
run (reproduced at all 3 error points).
