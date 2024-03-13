import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patchs = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels=channels, out_channels= embed_dim,
                              kernel_size=patch_size, stride=patch_size,bias=True
                              )
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features,hidden_features,bias=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
    
class MultiHead_Attention(nn.Module):
    def __init__(self, hidden_num, n_heads, dropout):
        super(MultiHead_Attention, self).__init__()
        self.n_heads = n_heads
        self.hidden_num = hidden_num
        self.head_dim = hidden_num // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_num, hidden_num*3,bias=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_num, hidden_num,bias=True)

    def forward(self,x):
        b, n, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2,-1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(1)
        attn = self.dropout(attn)

        w_avg = attn @ v
        w_avg = w_avg.transpose(1,2)
        w_avg = w_avg.flatten(2)

        x = self.proj(w_avg)
        x = self.dropout(x)
        return x
    
class transformer_block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, dropout):
        super(transformer_block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHead_Attention(hidden_num=dim,
                                        n_heads=n_heads,
                                        dropout=dropout
                                        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim,dropout=dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x
    
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, channels, n_classes, embed_dim, depth, n_heads, mlp_ratio, dropout):
        super(ViT, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patchs, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        self.transformer_block = nn.ModuleList(
            [
                transformer_block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.output = nn.Linear(embed_dim, n_classes,bias=True)
    def forward(self, x):
        n = x['image'].shape[0]
        x = self.patch_embed(x['image'])
        cls_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], 1)
        x += self.pos_embed
        x = self.dropout(x)

        for block in self.transformer_block:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        
        output = self.output(cls_token_final)
        return output