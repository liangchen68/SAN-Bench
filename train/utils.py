import numpy as np
import torch.nn.functional as F
import argparse
import torch
import os
import sys

def forward_pass(model,images,texts):
    image_features = model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)

    text_features = model.encode_text(texts)
    text_features = F.normalize(text_features, dim=-1)
    
    return image_features,text_features

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step <= warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length if steps != warmup_length else 1
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def check_and_save(args, model, optimizer, logs, epoch):
    if (epoch + 1) % args.save_every == 0:
        dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
        
        save_name = f"ckpt_{epoch+1}_{args.experiment_name}.pt"
        save_folder = os.path.join(args.ckpt_save_folder, args.experiment_name)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, save_name)
        torch.save(dump, save_path)

        print(f"Checkpoint saved at epoch: {epoch+1}!")
     

def save_at_end(args, model, optimizer, logs, epoch):
    last_ckpt_path = os.path.join(
        args.ckpt_save_folder, 
        args.experiment_name, 
        f"ckpt_{epoch+1}_{args.experiment_name}.pt")

    dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
    torch.save(dump, last_ckpt_path)

    logs_save_path = os.path.join(
        args.logs_save_folder,
        args.experiment_name,
        f"results_{args.experiment_name}.pt"
    )
    os.makedirs(os.path.join(args.logs_save_folder, args.experiment_name), exist_ok=True)
    torch.save(logs, logs_save_path)



def setup_args():
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument("--clip-model-name", type=str, default="path/to/clip") 
    parser.add_argument("--device", type=str, default="cuda")

    # training args
    parser.add_argument("--learning-rate", type=float, default=1e-6) 
    parser.add_argument("--norm-gradient-clip", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.2)

    
    parser.add_argument("--alpha", type=float) # this is lambda 2
    parser.add_argument("--beta", type=float) # lambda 1
   

    # data args
    parser.add_argument("--eval-dataset", type=str, default="caltech_101")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--type", type=int, default=2) # typically 2, do not need to change
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--early-exit", type=int, default=-1)
    parser.add_argument("--val-point", type=int, default=150)
    parser.add_argument("--negative-images", type=str, default="off")

    # save args
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--ckpt-save-folder", type=str, default="path/to/your/own/checkpoint_path")
    parser.add_argument("--logs-save-folder", type=str, default="path/to/your/own/checkpoint_path/logs")
    parser.add_argument("--experiment-name", type=str, default="SAN-CLIP")
    
    args = parser.parse_args()
    return args