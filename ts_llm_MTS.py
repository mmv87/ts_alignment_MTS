import torch
import torch.nn as nn
import torch.nn.functional as F
from  TS_encoder import PatchTSTEncoder
from  transformers import AutoModelForCausalLM,AutoTokenizer
from dataloader_MTS import ts_multimodal_text,collate_func
import os
from torch.utils.data import Dataset,DataLoader
device ='cuda' if torch.cuda.is_available() else 'cpu'

##loading the base LLM model and tokenizer
model_name='microsoft/Phi-4-mini-reasoning'
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
tokenizer =AutoTokenizer.from_pretrained(model_name,local_files_only=True)

device ='cuda' if torch.cuda.is_available() else 'cpu'
model_dtype=next(model.parameters()).dtype

## to expand the tokenizer to add the special tokens <ts> <ts/>
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))

##dataset fetching
import json
_json_file = os.path.join(os.environ["SLURM_TMPDIR"], "align.jsonl")

dataset= ts_multimodal_text(128,128,_json_file,tokenizer,device=device,model_dtype=None)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,max_patches,max_channel,patch_len,llm_model,device=device):
        super().__init__()
        self.tokenizer=tokenizer
        self.llm_model=llm_model
        self.embed_size=llm_model.config.hidden_size
        self.max_patches=max_patches
        self.max_channel=max_channel
        self.P=patch_len
        self.device=device

      ##initialise the ts_encoder
        ##self.ts_encoder=TS_Encoder_MLP(c_in=self.max_channel,N=self.max_patches,P=self.P,hidden_dim=512,output_dim=3072,shared_embedding=True)
        self.ts_encoder=PatchTSTEncoder(c_in=self.max_channel,num_patch=self.max_patches,patch_len=self.P,n_layers=2,d_model=self.embed_size,n_heads=4,shared_embedding=False,d_ff=2*256,
           norm='BatchNorm',attn_dropout=0,dropout=0.1,activation='gelu',store_attn=True,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=True)
        self.ts_encoder.to(self.device)
        
    def assemble_input_embeds(self,input_ids,ts_embeddings,ts_pairs:torch.tensor,labels:torch.tensor):
        assemb_embed_tensor=[]
        attention_mask_list=[]
        labels_list=[]
        assemb_labels=[]
        channels=ts_pairs.shape[1]
        
        ts_embeddings=ts_embeddings.view(-1,channels,10,self.embed_size)
        ts_embeddings.to(self.device)
        bs=ts_embeddings.shape[0]
        c_in=ts_embeddings.shape[1]
        num_ts_tokens=ts_embeddings.shape[2]
        ts_emb_dim=ts_embeddings.shape[3]
    
        input_embeds=self.llm_model.get_input_embeddings()(input_ids) ##seq_len,d_emb
        input_embeds.requires_grad_(requires_grad=True)
        text_emb_dim= input_embeds.shape[2]

        assert (ts_emb_dim==text_emb_dim)
    ##inplace operation to adjust ts_pairs 
        tokens_per_ts_inst = c_in * num_ts_tokens
    
        ts_pairs=ts_pairs.squeeze(0)
        ##print(f'original ts_pairs:{ts_pairs}')
        instance_indices = torch.arange(c_in, dtype=ts_pairs.dtype, device=self.device)
        cumulative_offsets = instance_indices * num_ts_tokens
        offset_expanded = cumulative_offsets.unsqueeze(1).expand(-1,2)

        ts_pairs_displaced=ts_pairs.to(device) + offset_expanded
        ts_pairs_displaced[:, 1] += num_ts_tokens
        flat_ts_embeddings=ts_embeddings.view(-1,c_in*num_ts_tokens,ts_emb_dim)
    
        if bs==1:
            flat_ts_embeddings=flat_ts_embeddings.squeeze(0) ##[c_in*ts_tokens,emb_dim]
            input_embeds=input_embeds.squeeze(0)
            ##print(f'flattened_text_embed : {input_embeds.shape}')
            ##print(f'flattened_ts_embed:{flat_ts_embeddings.shape}')

    ##total_seq_len = text_tokens + ts_tokens
        T_new = input_embeds.shape[0]+ c_in*num_ts_tokens
        local_indices= torch.arange(num_ts_tokens,device=device).repeat(c_in, 1)
    
        new_starts = ts_pairs_displaced[:,0] + 1
        new_starts.to(device)
        final_ts_indices = ((new_starts.unsqueeze(1).to(self.device)) + local_indices.to(self.device)).view(-1)
        is_ts_new=torch.zeros(T_new, dtype=torch.bool, device=self.device)
        is_ts_new[final_ts_indices]=True
    
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()
        text_container_new=torch.zeros((T_new,text_emb_dim),device=self.device)
        ts_container_new=torch.zeros((T_new,ts_emb_dim),device=self.device)

        ##scatter_ operation
        ts_scatter_idx=final_ts_indices.unsqueeze(1).expand(-1,ts_emb_dim)
        text_scatter_idx = new_text_indices.unsqueeze(1).expand(-1,text_emb_dim)

        text_container_new=text_container_new.scatter(dim=0,index=text_scatter_idx.to(device),src=input_embeds)
        ts_container_new=ts_container_new.scatter(dim=0,index=ts_scatter_idx.to(device),src=flat_ts_embeddings)
    
        final_embeds = text_container_new + ts_container_new
    
        assemb_embed_tensor.append(final_embeds)

        seq_len=final_embeds.shape[0]
        attention_mask=torch.ones((seq_len,),dtype=torch.long,device=self.device)
        attention_mask_list.append(attention_mask)

        ##return labels.
        loss_tokens=labels.squeeze(0).to(self.device)
        no_losstokens=torch.full((seq_len-labels.shape[1],),-100.0,dtype=torch.long,device=self.device)
        ##print(loss_tokens.shape,no_losstokens.shape)
        assemb_labels.append(torch.cat([no_losstokens,loss_tokens]))
        #print(torch.cat([no_losstokens,loss_tokens]))
        
        return torch.stack(assemb_embed_tensor),torch.stack(attention_mask_list,dim=0),torch.stack(assemb_labels,dim=0)
        
    

    def forward(self,input_ids=None,ts_input=None,ts_pairs=None,ts_masks=None,labels=None,):
        ##convert the ts_patches into ts_embeddings
        ts_tensor = ts_input.view(-1,self.max_patches,self.max_channel,self.P).to(self.device)  ## (bs,N,c_in,P)
        ts_embedding = self.ts_encoder(ts_tensor.to(self.device)) ## (bs,n_vars,num_patch,d_model)
        
        ##slicing
        ts_embedding_sliced =ts_embedding[ts_masks] ##flattened ts_embeddings
        input_embeddings,attentionmask_batch,lable_batch = self.assemble_input_embeds(input_ids,ts_embedding_sliced,ts_pairs,labels)
        
        attention_mask = attentionmask_batch.to(self.device)
        labels = lable_batch.to(self.device)
        output= self.llm_model(inputs_embeds=input_embeddings,attention_mask=attentionmask_batch,labels=lable_batch)
        
        return output,input_embeddings,ts_embedding_sliced
    
from tqdm import tqdm

model_wrapper=LLM_wrapper(tokenizer,10,20,128,model,device=device)
model_wrapper.train()
model_wrapper.to(device)

for p in model_wrapper.llm_model.parameters():
    p.requires_grad=False
for p in model_wrapper.llm_model.get_input_embeddings().parameters():
    p.requires_grad = True
    
for p in model_wrapper.ts_encoder.parameters():
    p.requires_grad = True
    
all_params = (list(model_wrapper.ts_encoder.parameters())+list(model_wrapper.llm_model.get_input_embeddings().parameters()))
optimizer = torch.optim.AdamW(all_params, lr=1e-5)
epoch_losses=[]

for epoch in range(1):  ##1 epochs
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    running_loss=0
    epoch_loss=0
    ctr=0
    for batch in pbar:
        input_ids=batch['input_ids'].to(device) ## input and output
        ts_input=batch['time_seried_padded'].to(device) ### batch of patchified padded ts_inputs (bs,c_in,N,p)
        ##print(ts_input.shape)
        ts_pairs=batch['ts_pairs']
        labels_batch=batch['labels'].to(device)
        ts_mask = batch['ts_mask'].to(device)
        
      ##model_wrapper=LLM_wrapper(tokenizer,ts_input,model,device=device)
        outputs,inputs,ts_embeds = model_wrapper(input_ids=input_ids,ts_input=ts_input,ts_pairs=ts_pairs,ts_masks=ts_mask,labels=labels_batch)
        loss=outputs.loss
        loss.backward()  ##gradient calculation

        running_loss+=loss.item()
        num_batches+=1
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())
        epoch_loss=running_loss/num_batches
        epoch_losses.append(epoch_loss)
        ctr+=1
        
        
### save the plot
out_path = os.path.join(os.environ["SLURM_TMPDIR"], "training_loss_MTS.png")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(8, 5))
plt.plot(ctr, epoch_losses, marker='o')
plt.title("Training Loss Trend Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(out_path)