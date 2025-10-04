from __future__ import annotations

from typing import Dict, Callable, Any, List
from datetime import datetime
import inspect
import yaml
import subprocess
import re

from response_parser import ResponseParser
from llm import LLM


class ReactAgent:
    """
    A minimal ReAct-style agent:
    - Maintains a simple message list (role, content, timestamp)
    - Uses a textual function-call format parsed by ResponseParser
    - Loops until a tool named `finish` is called or max_steps is reached
    - Registers a small set of tools provided externally
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Required attributes
        self.timestamp: int = int(datetime.utcnow().timestamp())
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1
        self.function_map: Dict[str, Callable[..., Any]] = {}

        # Internal state
        self._tests_unavailable: bool = False
        self._instructions: str = ""
        self._last_tool_name: str = ""
        self._last_tool_error_streak: int = 0

        # Minimal but strong system guidance for SWE-bench-style fixes
        self._system_prompt = (
            "ROLE: Bash-only coding agent fixing a single bug in a large repo.\n\n"
            "- Use ONLY the shell via run_bash_cmd for all actions (search, read, edit, test, diff).\n"
            "- Make the smallest viable change (1–5 lines). Never modify tests.\n"
            "- Do NOT narrate or describe changes. Do not say what you will do; just execute the commands to do it.\n"
            "- If stuck or repeating failures, call add_instructions_and_backtrack with a new strategy.\n\n"
            "WORKFLOW:\n"
            "1) Locate: run_bash_cmd('git grep -n \"KEY\" || true') or 'git ls-files | grep -n \"name\" || true'.\n"
            "2) Inspect: run_bash_cmd('nl -ba path/to/file.py | sed -n \"START,ENDp\" -n') or 'sed -n'.\n"
            "3) Edit: use standard shell techniques (e.g., heredoc to rewrite the file, or sed) via run_bash_cmd.\n"
            "   Example: echo new content to a temp file, then mv over the original; or use sed -i.\n"
            "4) Verify (right after an edit):\n"
            "   - run_bash_cmd('git status --porcelain && git --no-pager diff --unified=0 path/to/file.py')\n"
            "   - run_bash_cmd(\"sed -n 'START,ENDp' path/to/file.py\") to re-open the exact span\n"
            "   - If diff is empty or span unchanged: adjust anchors/lines and retry.\n"
            "   - If tests exist: prefer focused runs first (pytest -q -k KEY or single nodeid);\n"
            "     broaden only if the focused run passes.\n"
            "5) Finish only when git shows changes and behavior is validated.\n\n"
            "PITFALLS:\n"
            "- Wrong file path (e.g., 'postgres' vs 'postgresql'). Try variants if grep returns nothing.\n"
            "- Edits that don't land (wrong line range, no-op replacements). Always check git diff right after.\n"
            "- Unknown symbols: run_bash_cmd('git grep -n \"SYMBOL\" || true') before introducing new calls.\n"
            "- Text/bytes changes: run dual-path smoke with python -c for both bytes and str.\n"
            "- Docs/parsers: treat new warnings as failures and inspect output for 'WARNING'.\n"
            "- Masked test failures. Prefer explicit 'PASSED' checks or inspecting exit codes/output.\n\n"
            "RESPONSE RULE:\n"
            "Every response MUST end with exactly one function call using the required format.\n"
        )

        # Register built-in finish tool
        self.add_functions([self.finish, self.add_instructions_and_backtrack])

        # Seed system and user messages; user content is filled by run()
        self.root_message_id = self._add_message("system", "")
        self._user_message_index = self._add_message("user", "")

    # ---------------------------- Messaging ----------------------------
    def _add_message(self, role: str, content: str) -> int:
        parent_id = self.current_message_id
        msg_id = len(self.id_to_message)
        msg = {
            "role": role,
            "content": content,
            "timestamp": int(datetime.utcnow().timestamp()),
            "unique_id": msg_id,
            "parent": parent_id,
            "children": [],
        }
        self.id_to_message.append(msg)
        if parent_id != -1:
            self.id_to_message[parent_id]["children"].append(msg_id)
        self.current_message_id = msg_id
        if self.root_message_id == -1:
            self.root_message_id = msg_id
        return msg_id

    def _set_message(self, idx: int, content: str) -> None:
        self.id_to_message[idx]["content"] = content

    def _build_context(self) -> str:
        # Compose a minimal system block with available tools and the parser format
        tool_descriptions: List[str] = []
        for tool in self.function_map.values():
            signature = inspect.signature(tool)
            docstring = inspect.getdoc(tool) or ""
            tool_descriptions.append(
                f"Function: {tool.__name__}{signature}\n{docstring}\n"
            )
        tools_block = "\n".join(tool_descriptions)

        system_block = (
            f"{self._system_prompt}\n\n"
            f"--- AVAILABLE TOOLS ---\n{tools_block}\n\n"
            f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
        )

        # Walk from root to current using parent links to build the visible path
        visible_ids: List[int] = []
        cursor = self.current_message_id
        while cursor != -1:
            visible_ids.append(cursor)
            cursor = self.id_to_message[cursor]["parent"]
        visible_ids.reverse()

        transcript_parts: List[str] = [f"[system]\n{system_block}\n"]
        for mid in visible_ids:
            msg = self.id_to_message[mid]
            # Skip only the root system message (we already inject dynamic system block)
            if msg["role"] == "system" and mid == self.root_message_id:
                continue
            transcript_parts.append(f"[{msg['role']}]\n{msg['content']}\n")

        return "\n".join(transcript_parts)

    # ---------------------------- Tools ----------------------------
    def add_functions(self, tools: List[Callable[..., Any]]) -> None:
        for tool in tools:
            self.function_map[tool.__name__] = tool

    def finish(self, result: str) -> str:
        """
        Submit final result/summary. The runner will generate a git patch after
        this is returned. Keep the summary concise (what changed and why).
        """
        # Guard: only allow finish when there are code changes
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            )
            if status.returncode == 0 and not status.stdout.strip():
                raise ValueError(
                    "no_changes: No file modifications detected. Make a minimal edit before finishing."
                )
            # Extra guardrail: ensure there is a non-empty diff (staged or unstaged)
            staged = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True, timeout=5)
            unstaged = subprocess.run(["git", "diff"], capture_output=True, text=True, timeout=5)
            if (staged.returncode == 0 and unstaged.returncode == 0 and
                len((staged.stdout + unstaged.stdout).strip()) < 10):
                raise ValueError(
                    "empty_patch: Changes detected by status but no diff content. Ensure edits were written to disk."
                )
        except Exception:
            # If git is unavailable, fall back to allowing finish
            pass
        return result

    # ---------------------------- Helpers ----------------------------
    def _sanitize_arg(self, value: str) -> str:
        if not isinstance(value, str):
            return value
        # Remove stray argument markers accidentally included in values
        lines = [ln for ln in value.splitlines() if ln.strip() != "----END_ARGUMENT----"]
        return "\n".join(lines)

    def _normalize_response(self, text: str) -> str:
        """Make the function-call format more tolerant to minor typos.
        - Normalize any line starting with '----ARG' to the exact ARG separator
        - Collapse duplicated END markers
        """
        lines = text.splitlines()
        norm_lines: List[str] = []
        for ln in lines:
            if ln.strip().startswith("----ARG"):
                norm_lines.append("----ARG----")
            else:
                norm_lines.append(ln)
        norm_text = "\n".join(norm_lines)
        # Collapse duplicated END markers
        norm_text = re.sub(r"(----END_FUNCTION_CALL----)+", "----END_FUNCTION_CALL----", norm_text)
        return norm_text

    # Required API shims
    def set_user_prompt(self, user_prompt: str) -> None:
        self._set_message(self._user_message_index, user_prompt)

    def get_instructions(self) -> str:
        return self._instructions

    def set_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    def add_message(self, role: str, content: str) -> int:
        return self._add_message(role, content)

    def save_history(self, file_name: str) -> None:
        data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "root_message_id": self.root_message_id,
            "current_message_id": self.current_message_id,
            "id_to_message": self.id_to_message,
        }
        with open(file_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int) -> str:
        """
        Update instructions and move the current pointer back to an earlier message id.
        Messages are never removed; future steps will branch from the chosen node.
        """
        self._instructions = instructions
        try:
            at = int(at_message_id)
        except Exception:
            at = self._user_message_index
        if 0 <= at < len(self.id_to_message):
            self.current_message_id = at
            return f"Backtracked to message {at}."
        return "Invalid backtrack id; instructions updated."

    # ---------------------------- Main loop ----------------------------
    def run(self, task: str, max_steps: int) -> str:
        # Set the task as the user prompt
        self.set_user_prompt(task)

        for _ in range(max_steps):
            prompt = self._build_context()
            response = self.llm.generate(prompt)
            self._add_message("assistant", response)

            # Parse the tool call at the end
            try:
                parsed = self.parser.parse(self._normalize_response(response))
            except Exception as e:
                # Inject corrective system guidance and continue
                self._add_message(
                    "system",
                    (
                        "Format error: The response must end with exactly one function call using\n"
                        f"{self.parser.BEGIN_CALL} ... {self.parser.END_CALL}.\n"
                        "Respond again and include a tool call."
                    ),
                )
                continue

            tool_name = parsed["name"]
            # Sanitize argument values to remove stray parser markers
            args = {k: self._sanitize_arg(v) for k, v in parsed["arguments"].items()}

            if tool_name not in self.function_map:
                self._add_message(
                    "system", f"Unknown tool '{tool_name}'. Available: {list(self.function_map)}"
                )
                continue

            tool_fn = self.function_map[tool_name]
            try:
                result = tool_fn(**args)
            except Exception as e:
                hint = str(e)
                # Simple repetition detection and pivot via backtrack
                if tool_name == self._last_tool_name:
                    self._last_tool_error_streak += 1
                else:
                    self._last_tool_name = tool_name
                    self._last_tool_error_streak = 1
                if self._last_tool_error_streak >= 2:
                    instructions = (
                        "STUCK: Previous attempts failed. Pivot strategy:\n"
                        "1) Locate target with grep/ls-files (run_bash_cmd).\n"
                        "2) Inspect with show_file.\n"
                        "3) Apply minimal edit with replace_in_file.\n"
                        "4) Immediately verify the edit landed:\n"
                        "   - git status --porcelain && git --no-pager diff --unified=0 path/to/file\n"
                        "   - sed -n 'START,ENDp' path/to/file\n"
                        "5) If diff is empty or span unchanged, re-anchor (adjust window) and retry.\n"
                        "6) Then run a focused test (pytest -q -k KEY); broaden only if it passes.\n"
                        "Hint: If a path contains '/postgres/', try '/postgresql/'."
                    )
                    # Backtrack to user message to refresh context
                    self.add_instructions_and_backtrack(instructions, self._user_message_index)
                    self._add_message("system", f"AUTO-BACKTRACK applied due to repeated '{tool_name}' errors: {hint}")
                    # Reset streak after backtrack
                    self._last_tool_error_streak = 0
                    self._last_tool_name = ""
                    continue
                self._add_message("system", f"Tool '{tool_name}' error: {hint}")
                continue

            if tool_name == "finish":
                # reset streak after successful finish
                self._last_tool_error_streak = 0
                self._last_tool_name = ""
                return result

            # Record tool output to help the next step
            self._add_message("system", f"Result:\n{result}")
            # reset error streak on successful tool execution
            self._last_tool_error_streak = 0
            self._last_tool_name = ""

        return "Agent reached maximum steps without calling finish."


