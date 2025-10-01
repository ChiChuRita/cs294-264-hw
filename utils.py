import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from minisweagent import Environment
from minisweagent.environments import get_environment
import json
import threading
import subprocess

_OUTPUT_FILE_LOCK = threading.Lock()
    
def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name

def get_sb_environment(instance: dict) -> Environment:
    env_config = {
        "image": get_swebench_docker_image_name(instance),
        "cwd": "/testbed",
        "timeout": 60,
        "env": {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
        "environment_class": "docker",
    }
    env = get_environment(env_config)
    return env

def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def save_traj(
    agent: Any | None,
    path: Path,
    *,
    print_path: bool = True,
    result: str | None = None,
    **kwargs,
):
    """Save the trajectory of the agent to a file.

    Args:
        agent: The agent to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        result: The result/submission of the agent.
        **kwargs: Additional information to save (will be merged into top level)

    """
    data = {
        "info": {
            "submission": result,
        },
        "messages": [],
        "trajectory_format": "mini-swe-agent-1",
    } | kwargs
    if agent is not None:
        # NOTE: save messages if you want 
        data["info"]["config"] = {
            "agent": agent.name,
            "model": agent.llm.model_name,
        }
        
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    if print_path:
        print(f"Saved trajectory to '{path}'")