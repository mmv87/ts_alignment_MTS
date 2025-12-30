
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

    
class Transpose(nn.Module):
    def __init__(self, *dims,contiguous=False):
        super(Transpose, self).__init__()
        self.dims,self.contiguous= dims,contiguous
        
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
        
def positional_encoding(pe,learn_pe,q_len,d_model):
    if pe==None:
        W_pos=torch.empty((q_len,d_model))
        nn.init.uniform_(W_pos,-0.02,0.02)
        learn_pe=False ##flag to make the PE constant by setting flag to false 
    
    elif pe=='zero': ##scalar PE to learn the positions
        W_pos=torch.empty((q_len,1))
        nn.init.uniform_(W_pos,-0.02,0.02)
    
    elif pe=='zeros': # vectorial PE
        W_pos=torch.empty((q_len,d_model))
        nn.init.uniform_(W_pos,-0.02,0.02)
    
    elif pe=='normal' or 'gauss': ##scalar PE to learn the positions
        W_pos=torch.zeros((q_len,1))
        torch.nn.init.normal_(W_pos,mean=0.0,std=0.1)
        
    elif pe=='uniform':
        W_pos=torch.zeros((q_len,1))
        nn.init.uniform_(W_pos,a=0.0,b=1.0)
        
    return nn.Parameter(W_pos,requires_grad=learn_pe)     

## Time Series Encoder based on the transformer architecture
##MHSA- - Multi-head self attention
class MultiheadAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k=None,d_v=None,res_attention=False,
                 attn_dropout=0.1,proj_dropout=0.1,qkv_bias=True,lsa=False):
        
        super(MultiheadAttention, self).__init__()
        
        d_k=d_model//n_heads if d_k is None else d_k
        d_v=d_model//n_heads if d_v is None else d_v        
        self.n_heads,self.d_k,self.d_v=n_heads,d_k,d_v
        self.res_attention=res_attention   ## boolean to use attention
        
        self.W_Q=nn.Linear(d_model,self.d_k*n_heads,bias=qkv_bias)
        self.W_K=nn.Linear(d_model,self.d_k*n_heads,bias=qkv_bias)
        self.W_V=nn.Linear(d_model,self.d_v*n_heads,bias=qkv_bias)

        self.attention=ScaledDotProductAttention(d_model,n_heads,res_attention=res_attention,attn_dropout=attn_dropout,lsa=lsa)

        ##project to the output
        self.to_out= nn.Sequential(nn.Linear(n_heads*self.d_v,d_model),
                                   nn.Dropout(proj_dropout))
        
    def forward(self,Q:Tensor,K:Tensor,V:Tensor,mask=None):
        
        bs=Q.size(0)
        if K is None: K=Q
        if V is None:V=Q
        
        ##linear transformation of input tensor 'x' into Q,K and V
        q_s = self.W_Q(Q).view(bs,-1,self.n_heads,self.d_k).transpose(1,2) ##  (bs,n_heads,seq_len,d_k)
        k_s = self.W_K(K).view(bs,-1,self.n_heads,self.d_k).permute(0,2,3,1) ##  (bs,n_heads,seq_len,d_k)
        v_s = self.W_V(V).view(bs,-1,self.n_heads,self.d_v).transpose(1,2) ##  (bs,n_heads,seq_len,d_v)
        
        if self.res_attention:
            output,attn_scores,attn_weights= self.attention(q_s,k_s,v_s,mask=mask)        
        else:
            output,attn_weights=self.attention(q_s,k_s,v_s,mask=mask)
        ## output:[bs X n_heads X q_len X d_v]
        
        ## reassemble invidual heads to get MHSA output
        output= output.transpose(1,2).contiguous().view(bs,-1,self.n_heads*self.d_v) ## (bs,seq_len,n_heads*d_v) 
        
        output= self.to_out(output)
        
        if self.res_attention: 
            return output,attn_scores,attn_weights
        else: 
            return output,attn_weights 
##attention bloc/layer
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,n_heads,attn_dropout=0.1,res_attention=False,lsa=False):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout=nn.Dropout(attn_dropout)
        self.res_attention = res_attention  ## boolean to use attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa=lsa ## boolean to use learnable scale

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask=None):
        
        attn_scores =torch.matmul(q,k)*self.scale
        attn_weights =F.softmax(attn_scores, dim=-1)
        
        attn_weights= self.attn_dropout(attn_weights) ##regularization
        output = torch.matmul(attn_weights, v) ## contextually aware 'value' tensor 
        
        if self.res_attention:
            return output,attn_weights,attn_scores ## return the out in the order output,scores
        else:
            return output,attn_weights

##  Encoder layer [(MHSA + Norm) + (FFN +Norm)]
class TS_encoder_layer(nn.Module):
    def __init__(self, d_model, n_heads,d_ff=256, store_attn = False , 
                 norm= 'BatchNorm',attn_dropout=0,dropout=0.,bias=True,activation='gelu',res_attention=True,pre_norm=False):
        
        super(TS_encoder_layer, self).__init__()
        ##self.d_model=d_model

        assert not d_model%n_heads, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.mhsa = MultiheadAttention(d_model,n_heads,d_k=self.d_k,d_v=self.d_v,res_attention=res_attention,
                 attn_dropout=attn_dropout,proj_dropout=0.1,qkv_bias=True,lsa=False)
        
        self.res_attention = res_attention
        self.dropout_attn= nn.Dropout(dropout)

        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2),
                                            nn.BatchNorm1d(d_model),Transpose(1,2))
        
        else: 
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model,d_ff,bias=bias),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff,d_model,bias=bias))
        
        ##add & norm 
        self.dropout_ffn=nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2),
                                           nn.BatchNorm1d(d_model),Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
            
        self.pre_norm = pre_norm ## optional to normalize prior to encoder block
        self.store_attn = store_attn
        
        
    def forward(self, src:Tensor, prev=None, mask=None):
        ##multi-head attention 
        if self.pre_norm:
            src=self.norm_attn(src)
        
        ##Multi-Head attention
        if self.res_attention:
            src2,attn,scores = self.mhsa(src, src, src, prev)
        else:
            src2,attn = self.mhsa(src, src, src, mask=mask)    
        
        if self.store_attn:
            self.attn = attn
            
        ##Add & Norm
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src=self.norm_attn(src)

        ##Feed-forward layer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ##position-wise feed-forward
        src2 = self.ff(src)
        
        ##add & norm 
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src 

class TST_encoder(nn.Module):

    def __init__(self,d_model=None, n_heads=2,d_ff=256,norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu', n_layers=4, res_attention=False, pre_norm=False,store_attn=False):

        super(TST_encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TS_encoder_layer(d_model,n_heads,d_ff=d_ff,store_attn = store_attn,norm= norm,attn_dropout=attn_dropout,
                             dropout=dropout,bias=True,activation=activation,res_attention=res_attention,pre_norm=pre_norm)
            for _ in range(n_layers)
        ])

        self.res_attention = res_attention

    def forward(self, src: Tensor, prev=None, mask=None):
        output=src 
        scores=None
        
        if self.res_attention:
            for mod in self.layers:output,scores = mod(output, prev=scores, mask=mask)
            return output
        
        else:
            for mod in self.layers: output = mod(output)
            return output

class PatchTSTEncoder(nn.Module):
    def __init__(self,c_in=10,num_patch=10, patch_len=64,n_layers=3,d_model=128,n_heads=16,shared_embedding=False,d_ff=256,
                 norm='BatchNorm',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=False,**kwargs):
        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model=d_model  
        self.shared_embedding = shared_embedding ## bool to have channel independence/mixing
        self.activation=activation
        self.n_heads=n_heads
        
        #input_embedding
        if not shared_embedding: ###channel independent embedding
            self.W_P=nn.ModuleList()
            for _ in range(self.n_vars):self.W_P.append(nn.Linear(patch_len,self.d_model))
        else:
            self.W_P=nn.Linear(patch_len,self.d_model)
    
        ##positional encoding
        self.W_pos=positional_encoding(pe,learn_pe,num_patch,self.d_model) ## (num_patch,d_model)
        
        self.dropout=nn.Dropout(dropout)
        ##Encoder
        self.encoder = TST_encoder(d_model=self.d_model, n_heads=self.n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm,activation=self.activation,res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        
    def forward(self, x:Tensor, mask=None):
        bs,num_patch, n_vars, patch_len = x.shape ##the input tensor 'x' should be in the following order 
    
        if not self.shared_embedding:
            x_out =[]
            for i in range(n_vars):
                z=self.W_P[i](x[:,:,i,:]) ## (bs,num_patch,d_model)
                x_out.append(z)
                
            x=torch.stack(x_out,dim=2) ## (bs,N,c_in,d_model)
            
        else:
            x=self.W_P(x) ## (bs,N,c,d_model)
        
        x=x.transpose(1,2) ## (bs,d_model,num_patch)
        u=torch.reshape(x,(bs*n_vars,self.num_patch,self.d_model)) ## (bs*n_vars,num_patch,d_model)
        
        u=self.dropout(u+self.W_pos) ## dropout with positional encoding
        
        ## if only one patch, no need to pass through the encoder but with ts_padding tokens this becomes redundant
        if self.num_patch==1:
            return u.unsqueeze(0) ## (bs,num_patch,d_model)
        
        else:
            ##encoder
            z= self.encoder(u,mask=mask) ## (bs*n_vars,num_patch,d_model)
            z=torch.reshape(z,(-1,self.n_vars,num_patch,self.d_model)) ## reshaped to (bs,n_vars,num_patch,d_model)
            
            return z  ## (bs,n_vars,num_patch,d_model)
    