#!/usr/bin/env python3
"""
Quick test script to run the improved agent on just 5 strategic instances.
This is much faster than running all 20 instances.
"""
import concurrent.futures
from pathlib import Path

import typer
from datasets import load_dataset

from utils import save_traj, update_preds_file, remove_from_preds_file, get_sb_environment

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "cs294": "lynnliu030/swebench-eval-subset",
}

from agent import ReactAgent
from llm import OpenAIModel
from response_parser import ResponseParser
from envs import SWEEnvironment, DumbEnvironment

# Select 5 strategic instances to test improvements
TEST_INSTANCES = [
    "django__django-10973",    # Edge case: empty password handling
    "django__django-13297",    # Regression: template warning
    "psf__requests-1921",      # Regression: multivalued params
    "astropy__astropy-7166",   # Complete failure: deep understanding needed
    "sphinx-doc__sphinx-9230", # Complete failure: deep understanding needed
]

def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str,
    max_steps: int,
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
        env = SWEEnvironment(instance)
        # Initialize the agent
        agent = ReactAgent("swe-agent", parser, llm)
        
        # Add functions to the agent BEFORE running
        agent.add_functions([
            env.run_bash_cmd, 
            env.replace_in_file, 
            env.show_file, 
            agent.add_instructions_and_backtrack,
            env.search_in_file,
            env.list_functions,
            env.search_codebase,
            env.run_tests,
            env.search_and_replace,
            env.check_python_syntax,
        ])
        
        # Run the agent
        output = agent.run(task, max_steps) 
        
        # Generate patch for SWE-Bench
        result = env.generate_patch(output)
        
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

@app.command(help="Run improved agent on 5 test instances.")
def main(
    subset: str = typer.Option("cs294", "--subset", help="SWEBench subset used or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output: str = typer.Option("test_results", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model_name: str = typer.Option("gpt-5-mini", "--model", help="Model used", rich_help_panel="Basic"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum number of steps", rich_help_panel="Basic"),
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    all_instances = list(load_dataset(dataset_path, split=split))
    
    # Filter to just our test instances
    instances = [inst for inst in all_instances if inst["instance_id"] in TEST_INSTANCES]
    
    print(f"Testing on {len(instances)}/{len(all_instances)} instances:")
    for inst in instances:
        print(f"  - {inst['instance_id']}")
    print()

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_instance, instance, output_path, model_name, max_steps): instance[
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

