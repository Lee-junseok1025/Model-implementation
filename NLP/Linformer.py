import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Linformer(nn.Module):
    def __init__(self, 
                 embed_dim,
                 k_dim=256,
                 num_heads=8,
                 share_kv=False,
                 cross_attn=False
                ):
        super(DiffAttn, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        self.share_kv = share_kv
        self.cross_attn = cross_attn
        self.scale = self.head_dims ** -0.5
        

        self.q = nn.Linear(embed_dim,self.head_dims * self.num_heads, bias=False)

        kv_dim = self.head_dims if one_kv_head else (self.head_dims * self.num_heads) 
        self.k = nn.Linear(embed_dim,kv_dim,bias=False)
        self.v = nn.Linear(embed_dim,kv_dim,bias=False)
        self.out_proj = nn.Linear(self.head_dims * self.num_heads,embed_dim,bias=False)

        self.subln = nn.LayerNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
                  
    def attn(self,q,k,v,dropout_p):
      attn_w = F.softmax((q @ k.transpose(-1,-2)),dim=-1) * self.scale
      attn_w = F.dropout(attn_w,p=dropout_p)
      attn_score = attn_w @ v
      output = rearrange(attn_score,'b h d l -> b (h d) l')
      return output

    def forward(self,
                x,
                kv=None,
                kv_cache=None
        ):
          
          b, tgt_len, dim = x.size()
          q = self.q(x)
          if self.cross_attn is False or kv is None:
            k = self.k(x if kv is None else kv)
            if self.share_kv is False:
              v = self.v(x if kv is None else kv)
            else:
              v = self.k(x if kv is None else kv)
          else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]
          
          
          self.proj_k = nn.Parameter(init_(torch.zeros(tgt_len, self.k_dim)))
          self.proj_v = nn.Parameter(init_(torch.zeros(tgt_len, self.k_dim)))

          kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
          
          if not self.share_kv: # use k and v params
            k = torch.einsum('bld,lk -> bkl',k,self.proj_k)
            v = torch.einsum('bld,lk -> bkl',v,self.proj_v)
          else: # use only k params
            k = torch.einsum('bld,lk -> bkl',k,self.proj_k)
            v = torch.einsum('bld,lk -> bkl',v,self.proj_k)

          q = rearrange(q,'b l d -> b h d l',h=self.num_heads,d=self.head_dims)
          k = rearrange(k, 'b k l -> b h d l',h=self.num_heads,d=self.k_dim)
          v = rearrange(v, 'b k l -> b h d l',h=self.num_heads,d=self.k_dim)

          attn = self.attn(q,k,v)
          out = self.out_proj(attn)
          return out
if __name__ == '__main__':
    print('Evaluation')
    inputs = torch.randn(2,256,512) # Batch size tgt_len Embed dim
    attn = Linformer(embed_dim=512,k=256,num_heads=8)
    print(attn(inputs).shape)
