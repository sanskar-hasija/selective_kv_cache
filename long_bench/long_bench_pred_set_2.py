import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings 
warnings.filterwarnings("ignore")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,)
    parser.add_argument('--n_bits', type=int, required=True, help='Number of bits for quantization')
    parser.add_argument('--layer_strategy', type=str, required=True, help='Layer strategy for quantization')    
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    messages = [ 
        {
    "role": "system",
    "content":"""YYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""                
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(conversation = messages, add_generation_prompt=True, tokenize = False)
    
    return prompt

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map= "cuda:0"
    )
    model = model.eval()
    return model, tokenizer, model.device
def get_pred(data, max_length, max_gen, prompt_format, dataset, model_name, layer_strategy, n_bits):

    model, tokenizer,device = load_model_and_tokenizer(model_name)
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
    print("Dataset = ",dataset )
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue

            if layer_strategy != "zero":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    cache_implementation="quantized", 
                                 cache_config={
                                     "backend": "quanto", 
                                     "nbits": int(n_bits),
                                     "quantize_layers" : quantize_layers
                             }
            )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    use_cache = True
                )[0]
        else:
            if layer_strategy != "zero":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    cache_implementation="quantized", 
                     cache_config={
                                     "backend": "quanto", 
                                     "nbits": int(n_bits),
                                     "quantize_layers" : quantize_layers
                     }
                    )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache = True
                    )[0]
                
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    
    model_name = args.model
    layer_strategy = args.layer_strategy
    n_bits = args.n_bits
    # define your model
    max_length = 3500
    
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
        #             "dureader", "gov_report", "qmsum", "multi_news"]
        datasets = [ "vcsum", "trec", "triviaqa", "samsum", "lsht","passage_count", "passage_retrieval_en", 
                    "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # Load prompt formats and max generation lengths
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create directories for predictions
    os.makedirs("pred", exist_ok=True)
    os.makedirs("pred_e", exist_ok=True)
    
    # Predict on each dataset
    for dataset in tqdm(datasets):
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            os.makedirs(f"pred_e/{model_name}_{layer_strategy}_{n_bits}", exist_ok=True)
            out_path = f"pred_e/{model_name}_{layer_strategy}_{n_bits}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            os.makedirs(f"pred/{model_name}_{layer_strategy}_{n_bits}", exist_ok=True)
            out_path = f"pred/{model_name}_{layer_strategy}_{n_bits}/{dataset}.jsonl"
    
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        # Process all data samples
        get_pred(data, max_length, max_gen , prompt_format, dataset, model_name,layer_strategy,n_bits)
    

