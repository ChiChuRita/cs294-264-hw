#!/usr/bin/env python3
import concurrent.futures
from pathlib import Path

import typer
from datasets import load_dataset

from utils import save_traj, update_preds_file, remove_from_preds_file, get_sb_environment

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "cs294": "lynnliu030/swebench-eval-subset",
}

from simple_agent import ReactAgent
from llm import OpenAIModel
from response_parser import ResponseParser
from envs import MinimalSWEEnvironment

def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str,
    max_steps: int,
    baseline: bool,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    
    # Avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    
    # Initialize the model and parser
    llm = OpenAIModel(ResponseParser.END_CALL, model_name)
    parser = ResponseParser()
    task = instance["problem_statement"]
    
    print(f"Processing instance {instance_id}")
    agent = None    
    result = ""
    
    try:
        # Initialize the environment
        env = MinimalSWEEnvironment(instance)
        # Initialize the agent
        agent = ReactAgent("swe-react-agent", parser, llm)

        # Tool registration
        if baseline:
            # Bash-only baseline: expose only shell + backtrack
            agent.add_functions([
                env.run_bash_cmd,
                agent.add_instructions_and_backtrack,
            ])
            # Tighten the prompt to bash-only
            try:
                agent._system_prompt = (
                    "ROLE: Bash-only coding agent (baseline). Use ONLY run_bash_cmd for search, edit, test, and diff.\n\n"
                    "- Make the smallest viable change (1–5 lines). Never modify tests.\n"
                    "- After any edit, immediately verify with git status/diff and re-open the edited span.\n"
                    "- If stuck or repeating failures, use add_instructions_and_backtrack with a new strategy."
                )
            except Exception:
                pass
        else:
            # Minimal tools (bash-first) for the non-baseline agent
            agent.add_functions([
                env.run_bash_cmd,
                env.show_file,  # convenience wrapper around nl -ba
                env.replace_in_file,  # safe multi-line replace via heredoc
                agent.add_instructions_and_backtrack,
            ])
        
        # Run the agent (first pass)
        output = agent.run(task, max_steps)

        # Generate patch for SWE-Bench
        result = env.generate_patch(output)

        # Retry once if no valid patch was produced
        no_patch = not result.strip().startswith("diff --git") or "No changes detected" in result or "No file changes" in result
        if no_patch:
            try:
                retry_instructions = (
                    "RETRY: Previous attempt produced no patch. After each edit, immediately run:\n"
                    "- git status --porcelain && git --no-pager diff --unified=0 path/to/edited-file\n"
                    "- sed -n 'START,ENDp' path/to/edited-file\n"
                    "If diff is empty or span unchanged, re-anchor your window and try again.\n"
                    "Do NOT finish until git shows non-empty changes."
                )
                # Backtrack to the user message (id=1) with new instructions
                agent.add_instructions_and_backtrack(retry_instructions, 1)
                # Shorter second pass
                output_retry = agent.run(task, max_steps // 2)
                result = env.generate_patch(output_retry)
            except Exception as _:
                # Keep original result if retry fails
                pass
        
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        
    finally:
        # Save the trajectory and update the predictions file
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            result=result,
            instance_id=instance_id,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        print(f"Completed instance {instance_id}, result: {result}")

@app.command(help="Run CS294 HW on subset of SWEBench instances.")
def main(
    subset: str = typer.Option("cs294", "--subset", help="SWEBench subset used or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output: str = typer.Option("outputs", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model_name: str = typer.Option("gpt-5-mini", "--model", help="Model used", rich_help_panel="Basic"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum number of steps", rich_help_panel="Basic"),
    baseline: bool = typer.Option(False, "--baseline", help="Use bash-only baseline agent", rich_help_panel="Basic"),
    # NOTE: provide any extra arguments if needed
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    print(f"Running on {len(instances)} instances...")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")

    # Increase parallelism for faster benchmarking (adjust based on your CPU/RAM)
    # 20 workers = ~2x speedup, 30 workers = ~3x speedup (if you have resources)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_instance, instance, output_path, model_name, max_steps, baseline): instance[
                "instance_id"
            ]
            for instance in instances
        }
        try:
            process_futures(futures)
        except KeyboardInterrupt:
            print("Cancelling all pending jobs. Press ^C again to exit immediately.")
            for future in futures:
                if not future.running() and not future.done():
                    future.cancel()
            process_futures(futures)


if __name__ == "__main__":
    app()