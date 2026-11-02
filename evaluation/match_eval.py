import torch
from torch.utils.data import DataLoader
from train.dataset import EvalDataSet
from openclip import clip
import numpy as np

# load fine-tuned models. For CoN-CLIP, use ckpt, For CLIP(CC12M-NegFull) and NegCLIP,use ckpt['state_dict']
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path,map_location="cuda:2")
    model = model.float()
    model.load_state_dict(ckpt['model'])
    return model

model_path = '/path/to/clip'
device = 'cuda'
model, preprocess = clip.load("model_path", device=device)

check_point_path = "/path/to/fine-tuned models"
model = load_checkpoint(model, check_point_path) 

dataset = EvalDataSet(preprocess)
model = model.to(device)
bsz=16
loader = DataLoader(dataset, batch_size=bsz, num_workers=8, pin_memory=True, drop_last=False)
success=0
total=0
for i,batch in enumerate(loader):
    images,original_caption,negsub,negpred,negadv = batch
    # change it accordingly.
    # binary matching: texts = [original_caption,negsub]
    texts = [original_caption,negsub,negpred,negadv]
    texts = np.array(texts).T.flatten().tolist()
    total_texts = clip.tokenize(texts).to(device)
    images = images.to(device)
    image_features = model.encode_image(images)  
    text_features = model.encode_text(total_texts)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features = text_features.view(image_features.shape[0],-1,text_features.shape[-1])
    image_features = image_features.view(image_features.shape[0],1,-1)
    logits_per_image = model.logit_scale.exp() * image_features @ text_features.transpose(1,2)
    logits_per_image = logits_per_image.view(image_features.shape[0],-1)
    idx = torch.argmax(logits_per_image, axis=-1) 
    success+=(idx==0).sum().item()
    total+=image_features.shape[0]
    
print(success)
print(total)
print(success/total)