import argparse
import timeit
import torch
import torch.nn as nn
import tomllib
from cs336_basics import model
import os
import json

device="cuda" if torch.cuda.is_available() else "cpu"
assert device=="cuda","device must be GPU"

def parse_args():
    parser=argparse.ArgumentParser(
        description="Benchmarking Harness"
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small","medium","large","xl","10B"]
    )

    parser.add_argument(
            "--batch_size",
            type=int,
            default=4
        )

    parser.add_argument(
            "--context_length",
            type=int,
            default=512
        )

    parser.add_argument(
            "--test_stage",
            type=str,
            default="forward+backward+optimizer",
            choices=["forward-only","forward+backward","forward+backward+optimizer"]
        )

    parser.add_argument(
            "--precision_mode",
            type=str,
            default="fp32",
            choices=["fp32","bf16_mix"]
        )

    parser.add_argument(
            "--warmup_epoch",
            type=int,
            default=5
        )

    parser.add_argument(
            "--test_epoch",
            type=int,
            default=10
        )

    return parser.parse_args()

args=parse_args()

def get_model():
    current_dir=os.path.dirname(os.path.abspath(__file__))
    model_loading_path=os.path.join(current_dir,"config.toml")
    tokenizer_loading_path=os.path.join(current_dir, "tokenizer_config.toml")           
    with open(model_loading_path,"rb") as f:
        configs=tomllib.load(f)
    with open(tokenizer_loading_path,"rb") as f:
        tokenizer_configs=tomllib.load(f)
    model_config=configs[args.model_size]
    tokenizer_config=tokenizer_configs["base_config"]

    transformer_lm=model.BasicsTransformerLM(vocab_size=tokenizer_config["vocab_size"],
                                             context_length=args.context_length,
                                             d_model=model_config["d_model"],
                                             d_ff=model_config["d_ff"],
                                             num_layers=model_config["num_layers"],
                                             num_heads=model_config["num_heads"])
    return transformer_lm,tokenizer_config["vocab_size"]

def get_optimizer(model:nn.Module):
    return torch.optim.AdamW(model.parameters())


def forward_run_step(model:nn.Module,test_data:torch.Tensor):
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    start_time=timeit.default_timer()
    logit=model(test_data)
    torch.cuda.synchronize()
    end_time=timeit.default_timer()
    elapsed=end_time-start_time
    return elapsed

def backward_run_step(model:nn.Module,test_data:torch.Tensor):
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    start_time=timeit.default_timer()
    logit=model(test_data)
    logit=logit.sum()
    logit.backward()
    torch.cuda.synchronize()
    end_time=timeit.default_timer()
    elapsed=end_time-start_time
    return elapsed

def optimizer_run_step(model:nn.Module,test_data:torch.Tensor,optimizer:torch.optim):
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    start_time=timeit.default_timer()
    optimizer.zero_grad()
    logit=model(test_data)
    logit=logit.sum()
    logit.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end_time=timeit.default_timer()
    elapsed=end_time-start_time
    return elapsed

def save_metadata(save_dir,metadata:dict):
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,"benchmark.json")
    with open(save_path,"a") as f:
        f.write(json.dumps(metadata)+"\n")


def benchMarking():
    metadata={}
    elapsed_list=[]
    transformer_lm,vocab_size=get_model()
    transformer_lm=transformer_lm.to(device)
    if args.test_stage=="forward+backward+optimizer":
        optimizer=get_optimizer(transformer_lm)
    test_data=torch.randint(low=0,high=vocab_size,size=(args.batch_size,args.context_length),device=device)

    for _ in range(args.warmup_epoch):
        if args.test_stage=="forward-only":
            forward_run_step(transformer_lm,test_data)
        elif args.test_stage=="forward+backward":
            backward_run_step(transformer_lm,test_data)
        else:
            optimizer_run_step(transformer_lm,test_data,optimizer)


    for _ in range(args.test_epoch):
        if args.test_stage=="forward-only":
            elapsed=forward_run_step(transformer_lm,test_data)
        elif args.test_stage=="forward+backward":
            elapsed=backward_run_step(transformer_lm,test_data)
        else:
            elapsed=optimizer_run_step(transformer_lm,test_data,optimizer)
        elapsed_list.append(elapsed)

    mean=sum(elapsed_list)/args.test_epoch
    var=sum([(elapsed-mean)*(elapsed-mean) for elapsed in elapsed_list])/args.test_epoch
    std=var**0.5
    metadata[args.test_stage]=elapsed_list
    metadata["mean"]=mean
    metadata["std"]=std
    return metadata

if __name__=="__main__":
    metadata=benchMarking()
    save_dir="./output"
    save_metadata(save_dir,metadata)
    print(metadata)


   



    

    



    



    

    