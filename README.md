# CS294 HW1 - ReAct Agent for Software Engineering

This project implements a ReAct (Reasoning and Acting) agent for software engineering tasks using large language models. The codebase is adapted from [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/tree/main). 

## Installation

1. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```


## Implement the ReAct Agent
We provide a skeleton code for you to implement your ReAct agent. Refer to [CODE.md](./CODE.md) for more details about the structure of the code and what you should implement. 


## Running the Code

To run the ReAct agent on SWE-bench instances:
```bash
python run_agent.py --model gpt-5-mini --max-steps 100 --outputs results
```

The agent will process SWE-bench instances and save results to the `results/` directory.

**Note**: We suggest testing the agent on a single instance first by setting `instances = instances[:1]` in run_agent.py.


## Evaluation
### Running SWEBench's Evaluation Harness

After generating predictions, run SWEBench's evaluation harness to evaluate the submissions:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name lynnliu030/swebench-eval-subset \
    --predictions_path ./results/preds.json \
    --max_workers 8 \
    --run_id my_evaluation_run
```

## 📋 Evaluation Results Format

The evaluation will generate a results file with the following structure:

```json
{
    "total_instances": 20,
    "submitted_instances": 20,
    "completed_instances": 19,
    "resolved_instances": 9,
    "unresolved_instances": 10,
    "empty_patch_instances": 1,
    "error_instances": 0,
    "completed_ids": ["astropy__astropy-7166", ...],
    "resolved_ids": ["astropy__astropy-7166", ...],
    "unresolved_ids": ["django__django-10973", ...],
    "schema_version": 2
}
```

## 📤 Submission

After optimizing your agent, submit the following to the [submission server](http://vassar.millennium.berkeley.edu:8080/):

### 1. Code Artifact (ZIP)
- Must contain everything needed to build and run an end-to-end evaluation
- Do not commit secrets/keys
- Include setup instructions in README
- Ensure reproducible environment setup

### 2. Report (PDF)
Your report should:
- Report your accuracy number
- Describe the custom tools you created and explain the reason behind making them
- Share the lessons you learned

### 3. Final Evaluation Results (JSON)
The evaluation result file with the format shown above, containing your agent's performance metrics on the SWE-Bench subset.

---

*Good luck optimizing your SWE agent!* 🤖
