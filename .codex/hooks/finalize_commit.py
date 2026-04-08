#!/usr/bin/env python3
import json
import sys

data = json.load(sys.stdin)
prompt = (data.get("prompt") or "").lower()

trigger_terms = [
    "commit and push",
]

if any(term in prompt for term in trigger_terms):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": (
                        "This is a finalize-and-ship request. "
                        "Before committing or pushing, inspect the current diff and update "
                        "README.md and AGENTS.md if they should change based on the code changes."
                        "Make sure that AGENTS.md stays compact, so only include information if it is essential and delete unnecessary fat."
                        "AGENTS.md should contain information to make your futur life easier, so include information to not repeat the same mistakes."
                        "Then commit with a sensible message and push."
                    ),
                }
            }
        )
    )
else:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}))
