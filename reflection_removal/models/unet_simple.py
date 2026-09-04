import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class RMSnorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
    def forward(self,x):
        # F.normalize met la norme L2 des canaux a 1 ; * sqrt(dim) ramene le RMS a 1.
        # Sans elle il ne reste que le facteur sqrt(dim) : x8 puis x16 par Resblock,
        # les activations partent a 1e13 et fp16 deborde en NaN.
        return F.normalize(x, dim=1) * self.g * self.scale



class Resblock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # self.norm1 = nn.GroupNorm(8, out_ch) #batch norm
        self.norm1 = RMSnorm(out_ch) #we use RMSnorm which is more efficient
        self.act1 = nn.SiLU()
        if time_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(),nn.Linear(time_dim, 2 * out_ch)) # we're doing that for the scale and shift instead of just adding the time
        # self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # self.norm2 = nn.GroupNorm(8, out_ch) #batch norm
        self.norm2 = RMSnorm(out_ch)
        self.act2 = nn.SiLU()
        #res_conv = conv1x1 will be useful in upsampling
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity() #we need output and skip(x) to have the same dim
        # self.dropout = nn.Dropout(0.1) # let's see what it does
    def forward(self, x, t_emb=None):
        #block 1
        if t_emb is not None:
            t_context = self.time_proj(t_emb) #(B, 2 * out_ch)
            t_context = t_context[:,:,None,None] # We need the time_context to match output (B, 2 * out_ch, 1, 1)
            scale, shift = t_context.chunk(2, dim=1) #tuple (B, out_ch,1,1)
        
        h1 = self.conv1(x)
        h1 = self.norm1(h1)
        if t_emb is not None:
            h1 = h1 * (1 + scale) + shift #scale shift DONT FORGET TO TRY WHITHOUT SCALE adaGN
        h1 = self.act1(h1)
        #block 2
        # h1 = self.dropout(h1)
        h2 = self.conv2(h1)
        h2 = self.norm2(h2)
        h2 = self.act2(h2)
        return (h2 + self.res_conv(x))  # /np.sqrt(2) #adding variance normalisation with 1/sqrt(2)  

def Downsample(dima, dimb):
    return nn.Conv2d(dima, dimb, kernel_size=2, stride=2) # we divide by 2 the image feature resolution
def Upsample(dima, dimb):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"), #got helped by google on that one
        nn.Conv2d(dima, dimb, 3, padding=1),
    )
    

class Unet(nn.Module):
    def __init__(self,
                dim,# start dim of embedding in the Unet
                dim_mults=(2, 4),  # lets see
                channel=3  # input channel of our images :3
    ):
        super().__init__()
        self.dim = dim
        self.channel = channel
        self.init_conv = nn.Conv2d(channel, dim, kernel_size=3, padding=1) #we keep the same image_size kernel=3 padd=1
        
        #getting the dimensions for ups and downs
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:])) #giving couple of (in_ch, out_ch)
        
        #timesteps encoder
        # time_dim = dim*4 # we make the time embeddign big enough to be approx equal to feature embedding
        # self.t_embed = nn.Sequential(Sinusembed(self.dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        #ResBlocks downs
        self.downs = nn.ModuleList([])
        for (in_ch, out_ch) in in_out:
            down = nn.ModuleList([Resblock(in_ch, in_ch), Resblock(in_ch, in_ch), Downsample(in_ch, out_ch)])
            self.downs.append(down)
        
        #Midblocks
        mid_dim = dims[-1]
        self.mid_block1 = Resblock(mid_dim, mid_dim)
        self.mid_block2 = Resblock(mid_dim, mid_dim)

        # Resblock up
        self.ups = nn.ModuleList([])
        for (out_ch, in_ch) in reversed(in_out): #careful with the order in_ch, out_ch
            #we multiply by 2, cause we add the vector of the downresblocks as context
            up = nn.ModuleList([Upsample(in_ch, out_ch ),Resblock(2 * out_ch, out_ch), Resblock(2 * out_ch, out_ch)]) 
            self.ups.append(up)
        
        self.final_conv = nn.Conv2d(dim, channel, 1) #back to channel of the input

    def forward(self, x, t=None):
        if t is not None:
            t_emb = self.t_embed(t)
        else:
            t_emb=None
        inp = x
        x = self.init_conv(x)
        
        skips = []
        for down in self.downs:
            res1, res2, dsample = down
            x = res1(x, t_emb)
            skips.append(x)
            x = res2(x, t_emb)
            skips.append(x)
            x = dsample(x)
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        for up in self.ups:
            upsample, res1, res2 = up
            x = upsample(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = res1(x, t_emb)
            x = torch.cat((x, skips.pop()), dim=1)
            x = res2(x, t_emb)
        x = self.final_conv(x)
        return x + inp


        
