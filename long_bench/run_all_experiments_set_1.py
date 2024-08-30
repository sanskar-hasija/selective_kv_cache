import subprocess
import json
import pandas as pd
import os
import warnings
import time
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')


bits = [2,4]
layer_strategies = [ "zero", "all", "1/4_1", "1/4_2", "1/4_3", "1/4_4", "1/2_1", "1/2_2", "3/4_1", "3/4_2"]
models = ["meta-llama/Llama-2-7b-chat-hf"] 

def run_experiment(model_name, q_bits, layer_strategy):
    start_time = time.time()
    result = subprocess.run([
        "./run_experiment_long_bench_set_1.sh",
        "--model", model_name,
        "--n_bits", str(q_bits),
        "--layer_strategy", layer_strategy
    ], capture_output=False, text=True)
    end_time = time.time()
    return result

# Calculate total number of experiments
total_experiments = len(models) *  (len(layer_strategies) - 1) * len(bits) + len(models) 

# Run experiments
with tqdm(total=total_experiments, desc="Experiments Progress") as pbar:
    for layer_strategy in layer_strategies:
        if layer_strategy != "zero":
            for q_bit in bits:
                for model in models:
                    try:
                        print("Running experiment for model: ", model, " layer_strategy: ", layer_strategy, " q_bits: ", q_bit)
                        result = run_experiment(model, q_bit, layer_strategy)
                        if result:
                            print("Experiment finished")
                            print("-------------------------------------------------")
                    except Exception as E:
                        print(f"Error: {E}")
                pbar.update(1)
        else:
            for model in models:
                try:
                    print("Running experiment for model: ", model, " layer_strategy: ", layer_strategy)
                    result = run_experiment(model, 2, layer_strategy)
                    if result:
                        print("Experiment finished")
                        print("-------------------------------------------------")
                except Exception as E:
                    print(f"Error: {E}")
                pbar.update(1)