import argparse
import time
import itertools
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import QuantizedCacheConfig,QuantoQuantizedCache


def get_experiment_name(layer_strategy, n_bits):

    if layer_strategy != 'zero':
        experiment_name = 'experiment_' + layer_strategy.replace("/", "_")
        experiment_name = experiment_name + "_" + str(n_bits) + "_bits"
    else:
        experiment_name = 'experiment_' + layer_strategy
    return experiment_name
    

def get_quantized_layers(layer_strategy, model):
    if layer_strategy == "all":
        quantize_layers = [i for i in range(len(model.model.layers))]
    elif layer_strategy == "zero":
        quantize_layers = []
    elif layer_strategy == "1/4_1":
        quantize_layers = [i for i in range(0, len(model.model.layers)//4)]
    elif layer_strategy == "1/4_2":
        quantize_layers = [i for i in range(len(model.model.layers)//4, len(model.model.layers)//2)]
    elif layer_strategy == "1/4_3":
        quantize_layers = [i for i in range(len(model.model.layers)//2, 3*len(model.model.layers)//4)]
    elif layer_strategy == "1/4_4":
        quantize_layers = [i for i in range(3*len(model.model.layers)//4, len(model.model.layers))]
    elif layer_strategy == "1/2_1":
        quantize_layers = [i for i in range(0, len(model.model.layers)//2)]
    elif layer_strategy == "1/2_2":
        quantize_layers = [i for i in range(len(model.model.layers)//2, len(model.model.layers))]
    elif layer_strategy == "3/4_1":
        quantize_layers = [i for i in range(0, 3*len(model.model.layers)//4)]
    elif layer_strategy == "3/4_2":
        quantize_layers = [i for i in range(len(model.model.layers)//4, len(model.model.layers))]
    return quantize_layers

def compute_perplexity(
    model,
    tokenizer,
    dataset,
    experiment_name: str,
    layer_strategy: str,
    n_bits,
    output_dir: str = "outputs",
    data_column: str = "text",
    num_samples: Optional[int] = 1,
    num_tokens: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment_name}.csv"

    if output_file.exists() and not overwrite:
        raise ValueError(
            f"The {output_file!r} output file already exists - if you really want to override it, then use `--overwrite`."
        )

    logs = defaultdict(list)
    loss_fn = CrossEntropyLoss(reduction="none")
    quantize_layers = get_quantized_layers(layer_strategy, model)

    num_data_elements = 0
    for data_element in itertools.islice(dataset, num_samples):
        encodings = tokenizer(data_element[data_column], return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        pbar = tqdm(range(0, num_tokens - 1))
        num_processed_tokens = 0

        config = QuantizedCacheConfig(
            backend =  "quanto", 
            nbits = n_bits,
            quantize_layers = quantize_layers
        )
        if quantize_layers:
            past_key_values = QuantoQuantizedCache(config)
        else:
            past_key_values = None

        for idx in pbar:
            start_t = time.time()
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                perplexity = neg_log_likelihood.exp()
            pbar.set_description(f"nll: {neg_log_likelihood.item():>5.2f}, ppl: {perplexity.item():>8.2f}")

            # Store data and save every 10 tokens
            logs["data_idx"].append(num_data_elements + 1)
            logs["input_length"].append(idx + 1)
            logs["nll"].append(neg_log_likelihood.item())
            logs["ppl"].append(perplexity.item())
            logs["overall_ppl"].append(torch.tensor(logs["nll"]).mean().exp().item())
            logs["cuda_vram_allocated"].append(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)  # in GB
            logs["latency"].append(time.time() - start_t)
            if num_processed_tokens % 10 == 0:
                try:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                except KeyboardInterrupt as ex:
                    # If there's a Keyboard Interrupt, still write the file, and then stop
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                    raise ex

            num_processed_tokens += 1
            if num_tokens and num_processed_tokens >= num_tokens:
                break

        num_data_elements += 1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--layer_strategy", type=str, default='1/4_4')
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19-test")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=5000)
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="fp16")

    # Where to log
    parser.add_argument("--output_dir", type=str, default="experiments/")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="cuda:0",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Set up the dataset
    dataset = load_dataset(args.dataset_name, args.task, split=args.split, streaming=True)
    experiment_name = get_experiment_name(args.layer_strategy,  args.nbits)
    compute_perplexity(
        model,
        tokenizer,
        dataset,
        experiment_name,
        args.layer_strategy,
        args.nbits,
        output_dir=args.output_dir,
        data_column=args.data_column,
        num_samples=args.num_samples,
        num_tokens=args.num_tokens,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()