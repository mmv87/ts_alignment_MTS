from torch.utils.data import Dataset,DataLoader

from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import json

device ='cuda' if torch.cuda.is_available() else 'cpu'

class ts_multimodal_text(Dataset):
    def __init__(self,patch_len,stride,file_path,tokenizer,device=device,model_dtype=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.mode_dtype=model_dtype
        self.device=device
        self.p=patch_len
        self.s=stride
        self.offsets=[]
        
        with open(self.file_path,'rb') as f:
            offset=0
            for line in f:
                self.offsets.append(offset)
                offset+=len(line)
        
    def __len__(self):
        return len(self.offsets)
        
    ## to patchify/sliding window operation of the ts_input
    def padding_stride(self,x,p=256,s=256):
      x=x.view(1,-1)
      ##print(x.shape)
      l=x.shape[1]
      ##print(l)
      r =(l-p)%s
      if (r==0):
        num_windows=(l-p)//s+1
          #print(f'num_windows: {num_windows}')
        pad_width=0
        x_unfolded = x.unfold(1,p,s)
        return x_unfolded  ## (bs,N,P)

      else:
        num_windows=(l-p)//s+2
          #print(f'num_windows: {num_windows}')
        pad_width=s-r
        pattern = torch.tensor([0.0,1.0],device=self.device)
        num_repeats = pad_width // 2
        pad = pattern.repeat(num_repeats).view(1,-1)  ## (1,pad_width)
        x_padded = torch.cat([x,pad],axis=1)
        ##x_padded = torch.nn.functional.pad(x,(0,pad_width),mode='constant')
        x_unfolded = x_padded.unfold(1,p,s)
        return x_unfolded
          
    def parse_extract_ts_boundary(self,prompt):
        tokenized= self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        input_ids= tokenized['input_ids'][0]
        ts_start_token=self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token=self.tokenizer.convert_tokens_to_ids('<ts/>')
        ts_position=[]
    
        ##data structure to save the <ts>,<ts/> tokens ,list of tuples
        for i,token_id in enumerate(input_ids):
            if (token_id==ts_start_token):
                ts_position.append(('start',i))
            elif (token_id==ts_end_token):
                ts_position.append(('end',i))
                
        stack =[]
        ts_pairs=[]
        
        for j in range(len(ts_position)):
            pos,idx = ts_position[j]
            if pos=='start':
                stack.append(idx)
            elif stack and pos=='end':
                start=stack.pop(0)
                ts_pairs.append((start,idx))

        return ts_pairs,input_ids
        
    def __getitem__(self,idx):
        with open(self.file_path, 'rb') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line)
            
        ##sample=self.data[idx]
        prompt=sample['input']
        output=sample['output']
        timeseries=sample['timeseries']
        ts_inputs=[]

        output_ids=self.tokenizer(output,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        ts_pairs,prompt_ids=self.parse_extract_ts_boundary(prompt)
        ts_metrics=len(ts_pairs)
        ts_tensor=torch.tensor(timeseries,dtype=torch.float32)
        ##print(f'ts_tensor_size:{ts_tensor.shape}')

        ## loop through the n_vars
        for i in range(ts_metrics):
            ts_patched =self.padding_stride(ts_tensor[i,:],p=128,s=128)
            ts_patched=ts_patched.squeeze(dim=0)
            ##print(ts_patched.shape)

            if (ts_metrics==1):
                ts_inputs.append(ts_patched) ## (1,num_patches,1,patch_len)
            else:
                ts_inputs.append(ts_patched) ##(bs,N,C_in,P)

        combined_input_ids=torch.cat([prompt_ids,output_ids],dim=0)  ## input + output tokens
        
        return{ 'input_ids':combined_input_ids,
              'output_ids':output_ids,
              'ts_inputs':torch.stack(ts_inputs,dim=1),  ##stacking along the dim=1 (N,C_in,P)
              'ts_pairs':ts_pairs}   ##list of ts_data(tensors) of size (1,N_i,P)

###dataloader instantiation

import torch.nn.functional as F
### helper for collate function ts_mask for channel dimension

def mask(actual_in:torch.Tensor,device=device):
    ##c_in,N,P=ts_input.shape
    bs=1
    embed_dim=3072
    c_max=20
    N=10
    batched_c_in=actual_in.view(bs,1,1,1)
    channel_range = torch.arange(c_max).view(1,c_max,1,1) #bs,c_max,N,P
    mask = channel_range.to(device) < batched_c_in.to(device)
    token_mask=mask.expand(-1,-1,N,embed_dim)

    return token_mask

def collate_func(batch,tokenizer=None,device=device):
    input_ids = [x['input_ids'] for x in batch] ##(input+output_ids)
    output_ids=[x['output_ids'] for x in batch]
    ts_data=[x['ts_inputs'] for x in batch]  ###(N,C_in,P) list of tensors of shape(patches,n_vars,patch_len) for batch on inputs
    channels=torch.tensor([x['ts_inputs'].shape[1] for x in batch])    
    ts_patches_len=[x['ts_inputs'].shape[0] for x in batch] # Accessing shape from the actual tensor for univariate case
    ts_positions=[torch.tensor(x['ts_pairs']) for x in batch]

    ts_input_item=ts_data[0]
    
   ## setting the max_patch_length for ts_tokens
    max_n_per_batch=10
    max_channel_dim=20
    padded_ts_data=[]
    labels_batch=[]
    attention_mask_batch=[]
    
    ##padding the times series in the batch
    ## padding has to happen along the dim=0 with the padding values [0.0,1.0]
    ##padding along the dim=1 for channel dimension 
    for i,ts_input_sample in enumerate(ts_data):
        padded_patch_len=max_n_per_batch-ts_input_sample.shape[0]
        patch_len=ts_input_sample.shape[2]
        c_in=ts_input_sample.shape[1]
        ts_padding_len= padded_patch_len*patch_len*c_in
        pattern = torch.tensor([0.0,1.0]).to(device)
        num_repeats = ts_padding_len // 2
        pad = pattern.repeat(num_repeats)
    ##converting the ts_input = <ts_tokens>+<padded_ts_token>
        padded_ts_token=pad.view(-1,c_in,patch_len)
        padded_ts_data.append(torch.cat([ts_input_sample.to(device),padded_ts_token.to(device)],dim=0)) ## the ts_tokens are right padded

    ##pad along the channel dimension
    x=torch.stack(padded_ts_data)
    pad_channel=max_channel_dim-c_in
    pad=(0,0,0,pad_channel)
    ts_padded_channel = F.pad(x, pad, "constant", 0.0)
    
  ## N_i of batch of samples after padding
    ts_patch_padded_len=[x.shape[1] for x in padded_ts_data]
    max_text_len=max([x.size(0) for x in input_ids])
    max_ts_len=max(ts_patches_len)
    ts_seq_len = [seq.size(0) for seq in input_ids]
    tot_len=[(x+y) for x,y in zip(ts_patch_padded_len,ts_seq_len)]
    max_len_batch=max(tot_len)
    
##treat batch size of 1 as special case
    if len(batch)==1:
        input_ids_padded=input_ids[0].unsqueeze(0)    ##print(f'textual_shape {input_ids_padded.shape}')
        output_len=output_ids[0].shape[0]
        output_start_index = input_ids[0].shape[0] - output_len
        
        ##ts_mask for channel dimension
        ts_token=mask(channels,device=device)
        ## create a list of output_ids with no no_loss tokens
        labels_batch.append(output_ids)
        
        attention_mask=torch.cat([torch.ones(channels[0]*max_n_per_batch,dtype=torch.long,device=device),torch.ones(ts_seq_len[0],dtype=torch.long,device=device)])
        attention_mask_batch.append(attention_mask)

        return {
            'input_ids':input_ids_padded,
            "labels":torch.stack(output_ids),
            'attention_mask':torch.stack(attention_mask_batch),
            "time_seried_padded":ts_padded_channel,
            'ts_mask':ts_token,
            "time_series":torch.stack(padded_ts_data),
            'ts_pairs':torch.stack(ts_positions).to(dtype=torch.long)} ##list of tensor (N,C_in,P)}

    else:
        input_ids_padded= torch.stack([torch.cat([torch.full(((max_len_batch-seq.size(0)),),tokenizer.pad_token_id,dtype=seq.dtype),seq]) for seq in input_ids])

  ##max_len_batch=input_ids_padded.shape[1] # Correctepl=d to use shape[1] for sequence length
  ###max_N_per_batch=max(ts_data[])
  
    for i,sample in enumerate(batch):
        labels = torch.full((max_len_batch,),-100,dtype=torch.long,device=device)
        combined_len = sample['input_ids'].shape[0] + sample['ts_inputs'][0].shape[1] # Assuming one ts input per sample for simplicity
        pad_len = max_len_batch - combined_len
        seq_len=sample['input_ids'].shape[0]
        output_len=sample['output_ids'].shape[0]
        # Adjust label assignment based on padding at the beginning and TS embeddings
        # The labels correspond to the output_ids, which are at the end of the combined sequence
        # Calculate the starting index for output_ids in the padded label tensor
        output_start_index = max_len_batch - output_len
        labels[-output_len:] = sample['output_ids']
        labels_batch.append(labels)

        # Adjust attention mask based on padding and TS embeddings
        attention_mask=torch.cat([torch.zeros(pad_len,dtype=torch.long,device=device),torch.ones(sample['ts_inputs'][0].shape[1],dtype=torch.long,device=device),
                                torch.ones(seq_len,dtype=torch.long,device=device)
                                    ]) # Assuming one ts input
        attention_mask_batch.append(attention_mask)

  ##return the batch of input_ids , labels and timeseries
    return{
        'input_ids':input_ids_padded,
        "labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.cat(padded_ts_data,dim=0)} ##list of tensor (bs,max_N,Patch_len)