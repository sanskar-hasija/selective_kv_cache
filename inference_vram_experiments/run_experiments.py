import subprocess
import json
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define parameters for the experiments
bits = [2, 4]
output_tokens = [100, 1000, 2000, 4000, 10000, 20000]
layer_strategies = ["zero", "1/4", "1/2", "3/4", "all"]
models = ['meta-llama/Llama-2-7b-chat-hf']

# File to store experiment history
history_file = "experiment_history.csv"
results_file = "experiment_results.csv"

# Load existing history and results if available
if os.path.exists(history_file):
    history = pd.read_csv(history_file)
else:
    history = pd.DataFrame(columns=["model_name", "output_tokens", "layer_strategy", "q_bits"])

if os.path.exists(results_file):
    dataframe = pd.read_csv(results_file)
else:
    dataframe = pd.DataFrame(columns=["model_name", "output_tokens", "layer_strategy", "q_bits", "vram_consumption", "inference_time"])

def run_experiment(model_name, output_tokens, q_bits, layer_strategy):
    result = subprocess.run([
        "./run_experiment.sh",
        "--model_name", model_name,
        "--max_generation_length", str(output_tokens),
        "--n_bits", str(q_bits),
        "--layer_strategy", layer_strategy
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return None
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout

def experiment_completed(model, output_token, layer_strategy, q_bit):
    return not history[(history['model_name'] == model) & 
                       (history['output_tokens'] == output_token) & 
                       (history['layer_strategy'] == layer_strategy) & 
                       (history['q_bits'] == q_bit)].empty

def add_to_history(model, output_token, layer_strategy, q_bit):
    global history
    new_row = pd.DataFrame({
        "model_name": [model],
        "output_tokens": [output_token],
        "layer_strategy": [layer_strategy],
        "q_bits": [q_bit]
    })
    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(history_file, index=False)

def add_result(model, output_token, layer_strategy, q_bit, vram_consumption, inference_time):
    global dataframe
    new_row = pd.DataFrame({
        "model_name": [model],
        "output_tokens": [output_token],
        "layer_strategy": [layer_strategy],
        "q_bits": [q_bit],
        "vram_consumption": [vram_consumption],
        "inference_time": [inference_time]
    })
    dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    dataframe.to_csv(results_file, index=False)

# Calculate total number of experiments
total_experiments = len(models) * len(output_tokens) * (len(layer_strategies) - 1) * len(bits) + len(models) * len(output_tokens)

# Run experiments
with tqdm(total=total_experiments, desc="Experiments Progress") as pbar:
    for layer_strategy in layer_strategies:
        if layer_strategy != "zero":
            for q_bit in bits:
                for output_token in output_tokens:
                    for model in models:
                        if not experiment_completed(model, output_token, layer_strategy, q_bit):
                            try:
                                print("Running experiment for model: ", model," output_tokens: ", output_token, " layer_strategy: ", layer_strategy, " q_bits: ", q_bit)
                                result = run_experiment(model, output_token, q_bit, layer_strategy)
                                if result:
                                    print("Experiment finished")
                                    print("Inference Time: ", result['runtime'])
                                    print("VRAM consumption: ", result['vram_consumption'])
                                    print("-------------------------------------------------")
                                    add_result(model, output_token, layer_strategy, q_bit, result['vram_consumption'], result['runtime'])
                                else:
                                    add_result(model, output_token, layer_strategy, q_bit, "NA", "NA")
                            except Exception as E:
                                print(f"Error: {E}")
                                add_result(model, output_token, layer_strategy, q_bit, "NA", "NA")
                            add_to_history(model, output_token, layer_strategy, q_bit)
                        pbar.update(1)
        else:
            for output_token in output_tokens:
                for model in models:
                    if not experiment_completed(model, output_token, layer_strategy, "NA"):
                        try:
                            print("Running experiment for model: ", model, " output_tokens: ", output_token, " layer_strategy: ", layer_strategy)
                            result = run_experiment(model, output_token, 2, layer_strategy)
                            if result:
                                print("Experiment finished")
                                print("Inference Time: ", result['runtime'])
                                print("VRAM consumption: ", result['vram_consumption'])
                                print("-------------------------------------------------")
                                add_result(model, output_token, layer_strategy, "NA", result['vram_consumption'], result['runtime'])
                            else:
                                add_result(model, output_token, layer_strategy, "NA", "NA", "NA")
                        except Exception as E:
                            print(f"Error: {E}")
                            add_result(model, output_token, layer_strategy, "NA", "NA", "NA")
                        add_to_history(model, output_token, layer_strategy, "NA")
                    pbar.update(1)

print("All experiments completed. Results saved in", results_file)