import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Linformer(nn.Module):
    def __init__(self, 
                 embed_dim,
                 k_dim=256,
                 num_heads=8,
                 dropout_p=0.0,
                 share_kv=False,
                 cross_attn=False,
                 one_kv_head=False
                ):
        super(Linformer, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.k_h_dim = k_dim // num_heads
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        self.share_kv = share_kv
        self.cross_attn = cross_attn
        self.dropout_p = dropout_p
        self.scale = self.head_dims ** -0.5
        

        self.q = nn.Linear(embed_dim,self.head_dims * self.num_heads, bias=False)

        kv_dim = self.head_dims if one_kv_head else (self.head_dims * self.num_heads) 
        self.k = nn.Linear(embed_dim,kv_dim,bias=False) if not cross_attn else nn.Linear(embed_dim,self.head_dims * self.num_heads,bias=False)
        self.v = nn.Linear(embed_dim,kv_dim,bias=False) if not cross_attn else nn.Linear(embed_dim,self.head_dims * self.num_heads,bias=False)
        self.out_proj = nn.Linear(self.head_dims * self.num_heads,embed_dim,bias=False)
                  
    def attn(self,q,k,v,dropout_p):
      print(q.shape,k.shape)
      qk = (q * self.scale) @ (k * self.scale).transpose(-1,-2)
      attn_score = qk @ v
      output = rearrange(attn_score,'b h l d -> b l (h d)')
      return output

    def forward(self,
                x,
                kv=None,
                kv_cache=None
        ):
          
          b, tgt_len, dim = x.size()
          kv_len = tgt_len if kv is None else kv.shape[1]
          q = self.q(x)
          if not self.cross_attn or kv is not None:
              k = self.k(x if kv is None else kv)
              if self.share_kv is False:
                  v = self.v(x if kv is None else kv)
              else:
                  v = self.k(x if kv is None else kv)
          else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]
          

          proj_k_shape = (kv_len, self.k_dim) if self.cross_attn else (tgt_len, self.k_dim)
          self.proj_k = nn.Parameter(init_(torch.zeros(*proj_k_shape)))
          self.proj_v = nn.Parameter(init_(torch.zeros(*proj_k_shape)))

          if kv_len < tgt_len:
            self.proj_k = map(lambda t: t[:kv_len],self.proj_k)
            self.proj_v = map(lambda t: t[:kv_len],self.proj_v)

          kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
          
          if not self.share_kv and not self.cross_attn: # use k and v params
            k = torch.einsum('bld,lk -> blk',k,self.proj_k)
            v = torch.einsum('bld,lk -> blk',v,self.proj_v)

            k = rearrange(k, 'b (h d) l -> b h l d',h=self.num_heads,d=self.k_h_dim)
            v = rearrange(v, 'b (h d) l -> b h l d',h=self.num_heads,d=self.k_h_dim)
          elif self.cross_attn is True:
            k = rearrange(k, 'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
            v = rearrange(v, 'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
            print(k.shape)
          else: # use only k params
            k = torch.einsum('bld,lk -> blk',k,self.proj_k)
            v = torch.einsum('bld,lk -> blk',v,self.proj_k)
            k = rearrange(k, 'b (h d) l -> b h l d',h=self.num_heads,d=self.k_h_dim)
            v = rearrange(v, 'b (h d) l -> b h l d',h=self.num_heads,d=self.k_h_dim)
    
          q = rearrange(q,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
          
          print(q.shape,k.shape,v.shape)
          attn = self.attn(q,k,v,dropout_p=self.dropout_p)
          print(attn.shape)
          out = self.out_proj(attn)
          return out
if __name__ == '__main__':
    print('Evaluation')
    inputs = torch.randn(2,256,512) # Batch size tgt_len Embed dim
    input_2 = torch.randn(2,1024,512)
    attn = Linformer(embed_dim=512,k_dim=256,num_heads=8,cross_attn=True)
    print(attn(inputs).shape)
