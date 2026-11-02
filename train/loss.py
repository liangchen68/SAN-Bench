import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegLoss(nn.Module):
    def __init__(self, alpha ,beta) -> None:
        super().__init__()
        # loss weight
        self.alpha = alpha
        self.beta = beta
       
    def get_diag_labels(self,batch_size):
        labels = torch.arange(batch_size, dtype=torch.long)
        return labels
    
    def get_localhn_labels(self,batch_size):
        labels = torch.zeros(batch_size,dtype=torch.long)
        return labels
    
    # sigmoid loss labels
    def get_extra_labels(self,batch_size):
        eye_matrix = torch.eye(batch_size)
        labels = 1 - 2 * eye_matrix
        labels = labels.to(torch.long)
        return labels
 
    def forward(self,image_features,text_features,logit_scale):
        bsz = image_features.shape[0]
        labels = self.get_diag_labels(bsz).to(image_features.device)
        original_text_features = text_features[0:bsz]
        # base t2i and i2t loss
        t2i_logits = logit_scale * original_text_features @ image_features.T
        i2t_logits = t2i_logits.T

        extra_features = text_features[bsz:4*bsz].contiguous().view(bsz,-1,text_features.shape[-1])
        image_features = image_features.view(bsz,-1,image_features.shape[-1])
        image_to_text_logits_hn = logit_scale * image_features @ extra_features.transpose(1,2)
        image_to_text_logits_hn = image_to_text_logits_hn.contiguous().view(bsz,-1) 
        image_to_text_logits_hn = image_to_text_logits_hn[:,2] 
        image_to_text_logits_hn = image_to_text_logits_hn.unsqueeze(1)
        image_to_text_logits_with_hn = torch.cat([i2t_logits,image_to_text_logits_hn],dim=-1)
      
        # base_loss with hard negatives
        base_loss = (F.cross_entropy(t2i_logits,labels) + F.cross_entropy(image_to_text_logits_with_hn,labels))/2
        
        # intra-modal contrastive loss for T^pred and T^adv
        negpred_index = torch.arange(bsz+1,bsz*4,3)[:bsz]
        negpred_captions = text_features[negpred_index] 
        negpred_affirm_index = torch.arange(bsz*4,bsz*6,2)[:bsz]
        negpred_affirms = text_features[negpred_affirm_index] 
        
        negadv_index = torch.arange(bsz+2,bsz*4,3)[:bsz]
        negadv_captions = text_features[negadv_index]
        negadv_affirm_index = torch.arange(bsz*4+1,bsz*6,2)[:bsz]
        negadv_affirms = text_features[negadv_affirm_index] 
        
        i2i_logits_1 = logit_scale * negpred_captions @ negpred_affirms.T
        i2i_logits_2 = logit_scale * negadv_captions @ negadv_affirms.T
      
        ima_loss = (F.cross_entropy(i2i_logits_1,labels)+F.cross_entropy(i2i_logits_2,labels)+F.cross_entropy(i2i_logits_1.T,labels)+F.cross_entropy(i2i_logits_2.T,labels))/4
      
        # T^sub text features for sigmoid loss
        firsttype_neg_features = extra_features[:,0,:]
        i2t_logits_withonlyfisrtneg = logit_scale * image_features @ firsttype_neg_features.T  # 主对角线全是负样本 其余正样本
        i2t_logits_withonlyfisrtneg = i2t_logits_withonlyfisrtneg.view(bsz,-1)
        
        # ---------sigmoid loss for loss3------- 
        right_square = torch.full((bsz,bsz),1,dtype=torch.int32,device=image_features.device)
        right_square.fill_diagonal_(-1)
        gna_loss =  torch.log(1+torch.exp(right_square * (-i2t_logits_withonlyfisrtneg+30)))
        gna_loss = torch.sum(gna_loss)/bsz


        # total loss
        return base_loss + self.alpha * ima_loss + self.beta * gna_loss 
      