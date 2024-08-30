import subprocess
import json
import pandas as pd
import os
import warnings
import time
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')


models = [ 
    'Llama-2-7b-chat-hf_1_2_1_2',
    'Llama-2-7b-chat-hf_1_2_1_4',
    'Llama-2-7b-chat-hf_1_2_2_2',
    'Llama-2-7b-chat-hf_1_2_2_4',
    'Llama-2-7b-chat-hf_1_4_1_2',
    'Llama-2-7b-chat-hf_1_4_1_4',
    'Llama-2-7b-chat-hf_1_4_2_2',
    'Llama-2-7b-chat-hf_1_4_2_4',
    'Llama-2-7b-chat-hf_1_4_3_2',
    'Llama-2-7b-chat-hf_1_4_3_4',
    'Llama-2-7b-chat-hf_1_4_4_2',
    'Llama-2-7b-chat-hf_1_4_4_4',
    'Llama-2-7b-chat-hf_3_4_1_2',
    'Llama-2-7b-chat-hf_3_4_1_4',
    'Llama-2-7b-chat-hf_3_4_2_2',
    'Llama-2-7b-chat-hf_3_4_2_4',
    'Llama-2-7b-chat-hf_all_2',
    'Llama-2-7b-chat-hf_all_4',
    'Llama-2-7b-chat-hf_zero_2',
]


def run_experiment(model_name):
    start_time = time.time()
    result = subprocess.run([
        "./run_all_eval.sh",
        "--model", model_name,
    ], capture_output=False, text=True)
    end_time = time.time()
    return result

for model in models:
    print("Running evaluation for model: ", model)
    result = run_experiment(model)
    if result:
        print("Experiment finished")
        print("-------------------------------------------------")