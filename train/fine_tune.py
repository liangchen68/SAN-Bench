import torch
from torch.utils.data import DataLoader
from loss import HardNegLoss
from dataset import HardNegDataSet,HardNegDataWithAffirmSet
from utils import *
from utils import setup_args
from openclip import clip
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import math


def train_one_epoch(args,epoch,model,loader,optimizer,criterion, scheduler, scaler, early_exit=-1):

    autocast = torch.cuda.amp.autocast

    bar = tqdm(total=len(loader))
    running_loss = 0
  
    for i,batch in enumerate(loader):
        step = len(loader) * epoch + i
        scheduler(step) 
        # without intra-modal alignment
        if args.type == 1:
            images,original_caption,negsub,negpred,negadv = batch
            re = np.array([negsub,negpred,negadv]).T.flatten() 
            total_texts = original_caption + re.tolist() 
        else:  # with intra-modal alignment
            images,original_caption,negsub,negpred,negpred_affirm,negadv,negadv_affirm = batch
            re1 = np.array([negsub,negpred,negadv]).T.flatten()
            re2 = np.array([negpred_affirm,negadv_affirm]).T.flatten()           
            total_texts = original_caption + re1.tolist() + re2.tolist()
           
        total_texts = clip.tokenize(total_texts).to(args.device)
        images = images.to(args.device)

        optimizer.zero_grad()
        with autocast():
            image_features,text_features = forward_pass(model,images,total_texts)
            loss = criterion(image_features,text_features,model.logit_scale.exp())
            running_loss += loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            scaler.step(optimizer)
            scaler.update()      
        # backward        
        else:
            loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            optimizer.step()

        # clamp to 4.6052 = ln(100)
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100)) 

        # make logs after this
        avg_loss = running_loss / (i+1)
        logs = {"epoch": epoch+1, "avg_loss": avg_loss}
        bar.update(1)
        bar.set_postfix(logs)
        
        if early_exit == i:
            return logs 
    
    return logs
  
def train(args):

    # load model
    model, preprocess = clip.load(args.clip_model_name, device=args.device)
    model = model.float()
   # finetune the text-encoder
    for param in model.visual.parameters():
        param.requires_grad = False
    model.train()
   
    # get the dataset and loader
    if args.type==1:
         dataset = HardNegDataSet(preprocess)
    else: 
         dataset = HardNegDataWithAffirmSet(preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tqdm.write("dataset loaded")

    num_steps_per_epoch = len(loader)
    total_steps = args.epochs * num_steps_per_epoch

    criterion = HardNegLoss(args.alpha,args.beta)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    logs = {"fine_tuning": {}, "zero_shot_eval": {}}
    tqdm.write("Training start!")

    for epoch in range(args.epochs):
        # fine-tune every epoch and get logs
        training_logs = train_one_epoch(args, epoch, model, loader, optimizer, criterion, scheduler, scaler, early_exit=args.early_exit)
        # store logs
        logs["fine_tuning"][f"epoch_{epoch+1}"] = training_logs
        # checkpoint of (model, optimizer, logs) is stored every "args.save_every" epochs
        check_and_save(args, model, optimizer, logs, epoch)  
    save_at_end(args, model, optimizer, logs, epoch)
    tqdm.write("Training finished!")

train(setup_args())
