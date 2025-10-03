#!/usr/bin/env python3
"""
Quick script to test specific instances - much faster than full benchmark
Usage: python run_specific_instances.py instance1 instance2 ...
Example: python run_specific_instances.py django__django-13297 astropy__astropy-7166
"""

import sys
from pathlib import Path
from datasets import load_dataset
from run_agent import process_instance

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_specific_instances.py <instance_id1> [instance_id2] ...")
        print("\nExample quick tests:")
        print("  python run_specific_instances.py django__django-13297")
        print("  python run_specific_instances.py django__django-13297 astropy__astropy-7166")
        sys.exit(1)
    
    target_ids = set(sys.argv[1:])
    output_dir = Path("quick_test_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Testing {len(target_ids)} specific instance(s)...")
    print(f"Instances: {', '.join(target_ids)}\n")
    
    # Load dataset
    dataset = load_dataset("lynnliu030/swebench-eval-subset", split="test")
    
    # Filter to target instances
    instances = [inst for inst in dataset if inst["instance_id"] in target_ids]
    
    if len(instances) == 0:
        print(f"❌ No matching instances found!")
        print(f"Available instances:")
        for inst in dataset:
            print(f"  - {inst['instance_id']}")
        sys.exit(1)
    
    print(f"Found {len(instances)} instance(s)\n")
    
    # Run sequentially for easier debugging
    for i, instance in enumerate(instances, 1):
        instance_id = instance["instance_id"]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(instances)}] Processing: {instance_id}")
        print('='*60)
        
        try:
            process_instance(
                instance=instance,
                output_dir=output_dir,
                model_name="gpt-5-mini",
                max_steps=100  # Can reduce to 50 for even faster testing
            )
            print(f"✅ Completed: {instance_id}")
        except Exception as e:
            print(f"❌ Failed: {instance_id} - {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ All done! Results in: {output_dir}/")
    print('='*60)

if __name__ == "__main__":
    main()

