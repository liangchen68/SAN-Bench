from train.dataset import RetrievalSet
from openclip import clip
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# load fine-tuned models. For CoN-CLIP, use ckpt, For CLIP(CC12M-NegFull) and NegCLIP,use ckpt['state_dict']
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path,map_location="cuda")
    model = model.float()
    model.load_state_dict(ckpt["model"],strict=False)
    return model


model_path = '/path/to/clip'
model, preprocess = clip.load(model_path, device="cuda")
check_point = "path/to/fine-tuned models"
model = load_checkpoint(model, check_point) 
device = "cuda"
model=model.to(device)
dataset = RetrievalSet(preprocess)
bsz=32 # 
loader = DataLoader(dataset, batch_size=bsz, num_workers=8, pin_memory=True, drop_last=False)
total_acc = []
ranks =[]
for batch in tqdm(loader):

    images,original_caption,negsub,negpred,negadv = batch
    texts = original_caption + negsub + negpred + negadv
    total_texts = clip.tokenize(texts).to(device)
    images = torch.cat(images,dim=0).to(device)

    text_features = model.encode_text(total_texts)
    text_features = text_features.reshape(-1,len(original_caption),text_features.shape[-1])
    text_features = text_features.permute(1,0,2) 

    image_features = model.encode_image(images)  
    image_features = image_features.reshape(-1,len(original_caption),image_features.shape[-1])
    image_features = image_features.permute(1,0,2) 

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = model.logit_scale.exp() * text_features  @ image_features.transpose(1,2)
    rank = torch.argsort(logits,dim=-1,descending=True)
    ranks.append(rank)

def calculate_accuracies(data: torch.Tensor):

    data = data.float()
    bsz = data.shape[0]

    # Aff accuracy
    row1_val = data[:, 0, 0]
    acc1 = (row1_val == 0).float().mean()

    # Sub accuracy
    row2_vals = data[:, 1, 0:3]  
    has_one = (row2_vals == 1).any(dim=1)
    has_two = (row2_vals == 2).any(dim=1)
    has_three = (row2_vals == 3).any(dim=1)
    is_successful = has_one & has_two & has_three
    success_count = is_successful.sum().item()
    acc2 = success_count / bsz
    # Pred accuracy
    row3_vals = data[:, 2, 0:2]
    has_two = (row3_vals == 2).any(dim=1)
    has_three = (row3_vals == 3).any(dim=1)
    is_successful =  has_two & has_three
    success_count = is_successful.sum().item()
    acc3 = success_count / bsz
    
    # Adv accuracy
    row4_val = data[:, 3, 0]
    acc4 = (row4_val == 3).float().mean()
     
    return {
        "Type 1 Accuracy": acc1.item(),
        "Type 2 Accuracy": acc2,
        "Type 3 Accuracy": acc3,
        "Type 4 Accuracy": acc4.item(),
    }


accuracy = calculate_accuracies(torch.cat(ranks,dim=0))
print(accuracy)
    










