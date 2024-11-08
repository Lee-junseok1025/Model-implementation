import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from flash_attn import flash_attn_func

# from apex.apex.normalization import FusedRMSNorm

def set_lambda(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class DiffAttn(nn.Module):
    def __init__(self, embed_dim,depth,num_heads,flash_attn=False):
        super(DiffAttn, self).__init__()
        self.flash_attn = flash_attn
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim,embed_dim,bias=False)
        self.k = nn.Linear(embed_dim,embed_dim//self.n_rep,bias=False)
        self.v = nn.Linear(embed_dim,embed_dim//self.n_rep,bias=False)
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=False)


        self.lambda_init_ = set_lambda(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = nn.LayerNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self,
                x,
                mask=None):
        
        b, tgt_len, dim = x.size()
        src_len = tgt_len
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(b, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(b, src_len, 2 * self.num_kv_heads, self.head_dim)


        offset = tgt_len - src_len

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2 ,dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init_

        if self.flash_attn is True:
            q = q.reshape(b, tgt_len, self.num_heads, 2, self.head_dim)
            k = k.reshape(b, src_len, self.num_kv_heads, 2, self.head_dim)
            v = v.view(b, src_len, self.num_kv_heads, 2, self.head_dim)
            
            # chunk q,k,v 
            q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
            k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
            v1, v2 = v[:, :, :, 0], v[:, :, :, 1]
            attn1_1 = flash_attn_func(q1,k1,v1,causal=True)
            attn1_2 = flash_attn_func(q1,k1,v2,causal=True)
            attn1 = torch.cat([attn1_1,attn1_2],dim=-1)

            attn2_1 = flash_attn_func(q2,k2,v1,causal=True)
            attn2_2 = flash_attn_func(q2,k2,v2,causal=True)
            attn2 = torch.cat([attn2_1,attn2_2],dim=-1)
            attn = attn1 - lambda_full * attn2

        else:
            q = q.transpose(1,2) * self.scale
            k = repeat_kv(k.transpose(1,2), self.n_rep)
            v = v.view(b, src_len, self.num_kv_heads, 2 * self.head_dim)
            v = repeat_kv(v.transpose(1,2), self.n_rep)
            print(q.shape,k.shape,v.shape)
            attn = q @ k.transpose(-1,-2)
            if mask is None:
                attn_mask = torch.triu(
                    torch.zeros([tgt_len,src_len])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(attn),
                    1 + offset,
                )
            attn = torch.nan_to_num(attn)
            attn += attn_mask
            attn = F.softmax(attn,dim=-1,dtype=torch.float32).type_as(attn)
            print(attn.shape)
            attn = attn.view(b, self.num_heads, 2, tgt_len, src_len)
            print(attn.shape)
            # attn = rearrange(attn,'b (h two) tgt_len src_len -> b h two tgt_len src_len',two=2,h=self.num_heads)
            print( attn[:, :, 0].shape)
            attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]

            print(attn.shape,v.shape)
            attn = attn @ v
        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init_)
        attn = attn.reshape(b, tgt_len, self.num_heads * 2 * self.head_dim)
        # attn = rearrange(attn.transpose(1,2),'b l h d -> b l (h d)')

        attn = self.out_proj(attn)
        return attn



if __name__ == '__main__':
    print('Evaluation')
    inputs = torch.randn(2,256,512)
    attn = DiffAttn(embed_dim=512,depth=3,num_heads=8,flash_attn=False)
    print(attn(inputs).shape)





