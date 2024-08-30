import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess as sp
import time
import warnings
import argparse
import json

warnings.filterwarnings('ignore')

class TorchTracemalloc():
    track_memory_consumption = []
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        peak = torch.cuda.max_memory_allocated()
        peaked = (peak - self.begin) // 1024 ** 2
        TorchTracemalloc.track_memory_consumption.append(peaked)

def run_experiment(model_name, max_generation_length, n_bits, layer_strategy):
    with open('input_text.txt', 'r') as file:
        input_text = file.read()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name != "microsoft/Phi-3-medium-128k-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map= "cuda:0"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map= "auto"
        )
    
    


    if layer_strategy == "all":
        quantize_layers = [i for i in range(len(model.model.layers))]
    elif layer_strategy == "zero":
        quantize_layers = []
    elif layer_strategy == "1/4":
        quantize_layers = [i for i in range(0, len(model.model.layers)//4)]
    elif layer_strategy == "1/2":
        quantize_layers = [i for i in range(0, len(model.model.layers)// 2)]
    elif layer_strategy == "3/4":
        quantize_layers = [i for i in range(0, 3 * len(model.model.layers)//4)]

    if not quantize_layers:
        start = time.time()
        with TorchTracemalloc() as tt:
            inputs = tokenizer(input_text, max_length=100, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            out = model.generate(**inputs, 
                                 do_sample=False, 
                                 eos_token_id = None, 
                                 max_new_tokens=max_generation_length, 
                                 use_cache=True
                                )
            
            del out
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        runtime = time.time() - start
        vram_consumption = TorchTracemalloc.track_memory_consumption[-1]
        

    else:
        start = time.time()
        with TorchTracemalloc() as tt:
            inputs = tokenizer(input_text, max_length=100, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            out = model.generate(**inputs, 
                             do_sample=False, 
                             eos_token_id = None, 
                             max_new_tokens=max_generation_length, 
                             cache_implementation="quantized", 
                             cache_config={
                                 "backend": "quanto", 
                                 "nbits": n_bits,
                                 "quantize_layers" : quantize_layers
                             }
                            )
            del out
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        runtime = time.time() - start
        vram_consumption = TorchTracemalloc.track_memory_consumption[-1]

    return {"runtime": runtime, "vram_consumption": vram_consumption}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model generation script with quantization.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--max_generation_length', type=int, required=True, help='Maximum generation length')
    parser.add_argument('--n_bits', type=int, required=True, help='Number of bits for quantization')
    parser.add_argument('--layer_strategy', type=str, required=True, help='Layer strategy for quantization')
    
    args = parser.parse_args()

    result = run_experiment(args.model_name, args.max_generation_length, args.n_bits, args.layer_strategy)
    print(json.dumps(result))