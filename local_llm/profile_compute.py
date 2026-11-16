"""Profile compute resources for local LLM normalization."""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch
from local_llm.client import OllamaClient
from local_llm.compute_profiler import profile_model_inference, ComputeProfiler, BatchProfiler
from local_llm.normalize import load_config, LocalLLMNormalizer, build_prompt
from preprocessing.dataset import load_train_pairs
import json


def main():
    # Load config
    config_path = os.path.join(HERE, "configs", "qwen2.5_3b.yaml")
    config = load_config(config_path)
    
    # Initialize normalizer
    normalizer = LocalLLMNormalizer(config)
    
    # Load test queries
    data_dir = os.path.join(ROOT, "data")
    tp = load_train_pairs(data_dir).set_index("query_id")
    
    with open(os.path.join(ROOT, "methods", "cache", "splits.json")) as f:
        splits = json.load(f)
    
    qids = splits["test"][:20]  # Profile 20 queries
    informal_queries = [tp.at[q, "generated_informal_query"] for q in qids]
    
    # Select few-shot examples
    import random
    random.seed(42)
    train_qids = splits["train"]
    shot_qids = random.sample(train_qids, min(config.few_shot_examples, len(train_qids)))
    shots = [(tp.at[q, "generated_informal_query"], tp.at[q, "formal_query"]) for q in shot_qids]
    
    # Prepare prompts
    prompts = [build_prompt(q, shots) for q in informal_queries]
    
    print(f"Profiling {config.model_name} on {len(prompts)} queries...")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Profile using the client directly
    client = normalizer.client
    
    def generate_fn(prompt):
        return client.generate(prompt, options=normalizer.options)
    
    # Run profiling
    summary = profile_model_inference(
        model_name=config.model_name,
        model_fn=generate_fn,
        inputs=prompts,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=1,
        warmup=3,
        track_power=True,
        output_path=os.path.join(HERE, "cache", "compute_profile.json"),
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()