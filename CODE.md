## CS294 HW1 Starter: Minimal ReAct Agent Scaffold

This folder contains a scaffold you must complete to build a minimal ReAct agent for SWE-Bench.

You will:
- Maintain a message history tree (system → user → instruction, plus subsequent nodes)
- Implement a textual function-call protocol (no JSON/XML) and parse it with rfind
- Add tools (inside the agent): `finish` (implemented), `add_instructions_and_backtrack`
- Add tools (inside the environment): `run_bash_cmd` (implemented), and more you need to add 
- Run on SWE-Bench verified subset; report baseline and improved accuracy

### Entry point 
The entry point to the code is `run_agent.py`. It wires up the model, parser, agent, and environment, and provides the CLI. 
You don't need to modify this code. 

### Files to complete
- `agent.py` — Implement main logic of ReAct agent (e.g., message tree, tools, and main loop)
- `llm.py` — Implement `OpenAIModel` (or your backend) with `generate(prompt)`
- `env.py` - Implement additional functions to run inside the environment 
- `response_parser.py` — Implement `ResponseParser.parse` using rfind

### Message history tree (required)
Each message is a dict with:
- `role`: "system" | "user" | "assistant" | "tool" | "instructor" ...
- `content`: string (can be Markdown)
- `timestamp`: int (creation time)
- `unique_id`: int counter starting at 1 (or 0), unique per message
- `parent`: unique_id of parent (root has no parent)
- `children`: list of child unique_ids

Root path must be: system → user → instructor. The system prompt content must begin with: "You are a Smart ReAct agent." and automatically include:
- List of available tools (signature + docstring)
- Response format description (see `ResponseParser.response_format`)

### Function-call protocol (required)
LLM must output a single function call at the end:
```
your_thoughts_here
...
----BEGIN_FUNCTION_CALL----
function_name
----ARG----
arg_name
arg_value
...
----END_FUNCTION_CALL----
```
Parse using `str.rfind` to avoid issues with earlier markers.

### Required tools
- `run_bash_cmd(command: str)` — Execute a shell command via your SWE environment
- `finish(result: str)` — Finalize and return result string from `agent.run`
- `add_instructions_and_backtrack(instructions: str, at_message_id: int)` — Update instruction node content and move current pointer to `at_message_id`

### Limits and evaluation
- Backend model: GPT-5 mini (medium reasoning)
- `MAX_STEPS` must be capped at 100
- Baseline: report accuracy without `add_instructions_and_backtrack`
- Improved: report accuracy with `add_instructions_and_backtrack` and any more custom tools

### Setup
1) Install dependencies
```bash
uv pip install -r requirements.txt
```

2) Configure your API key via environment or `.env` (implementation-dependent)

### Run (single instance scaffold)
```bash
python starter/run_agent.py --instance-id my_instance --model gpt-5-mini --max-steps 100
```

For full evaluation, follow the course README/evaluation harness instructions and produce the JSON results specified in the assignment.
