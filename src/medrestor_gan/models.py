
import torch, torch.nn as nn, torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()
        self.c = nn.Conv2d(in_ch,out_ch,k,s,p,bias=not norm)
        self.n = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.a = nn.LeakyReLU(0.2, inplace=True)
    def forward(self,x): return self.a(self.n(self.c(x)))
class ResidualBlock(nn.Module):
    def __init__(self,ch): super().__init__(); self.b1=ConvBlock(ch,ch); self.b2=ConvBlock(ch,ch)
    def forward(self,x): return x + self.b2(self.b1(x))
class SelfAttention(nn.Module):
    def __init__(self,c): super().__init__(); self.q=nn.Conv2d(c,c//8,1); self.k=nn.Conv2d(c,c//8,1); self.v=nn.Conv2d(c,c,1); self.g=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B,C,H,W=x.shape; q=self.q(x).view(B,-1,H*W); k=self.k(x).view(B,-1,H*W); v=self.v(x).view(B,C,H*W)
        att=(q.permute(0,2,1)@k)/((k.shape[1])**0.5); att=att.softmax(-1); out=(v@att.permute(0,2,1)).view(B,C,H,W)
        return self.g*out + x
class Down(nn.Module):
    def __init__(self,i,o,attn=False): super().__init__(); self.b=nn.Sequential(ConvBlock(i,o,4,2,1),ConvBlock(o,o),ResidualBlock(o)); self.a=SelfAttention(o) if attn else nn.Identity()
    def forward(self,x): x=self.b(x); return self.a(x)
class Up(nn.Module):
    def __init__(self,i,skip,o,attn=False): super().__init__(); self.u=nn.ConvTranspose2d(i,o,4,2,1); self.b=nn.Sequential(ConvBlock(o+skip,o),ResidualBlock(o)); self.a=SelfAttention(o) if attn else nn.Identity()
    def forward(self,x,s): x=self.u(x); x=torch.cat([x,s],1); x=self.b(x); return self.a(x)
class UNetGenerator(nn.Module):
    def __init__(self,in_ch=2,base=64): 
        super().__init__()
        self.d1=Down(in_ch,base); self.d2=Down(base,base*2); self.d3=Down(base*2,base*4,attn=True); self.d4=Down(base*4,base*8); self.d5=Down(base*8,base*8,attn=True)
        self.bot=nn.Sequential(ConvBlock(base*8,base*8),ResidualBlock(base*8),ResidualBlock(base*8))
        self.u5=Up(base*8,base*8,base*8,attn=True); self.u4=Up(base*8,base*4,base*4); self.u3=Up(base*4,base*2,base*2,attn=True); self.u2=Up(base*2,base,base); self.u1=Up(base,base,base//2)
        self.out=nn.Conv2d(base//2,1,3,1,1)
    def forward(self,x):
        s1=self.d1(x); s2=self.d2(s1); s3=self.d3(s2); s4=self.d4(s3); s5=self.d5(s4); b=self.bot(s5)
        x=self.u5(b,s4); x=self.u4(x,s3); x=self.u3(x,s2); x=self.u2(x,s1); x=self.u1(x, x.new_zeros(x.shape)); return torch.tanh(self.out(x))*0.5+0.5
def ConvD(i,o,k=4,s=2,p=1,norm=True):
    L=[nn.Conv2d(i,o,k,s,p)]; 
    if norm: L.append(nn.InstanceNorm2d(o)); 
    L.append(nn.LeakyReLU(0.2,inplace=True)); 
    return nn.Sequential(*L)
class PatchDiscriminator(nn.Module):
    def __init__(self,in_ch=2,base=64): super().__init__(); self.net=nn.Sequential(ConvD(in_ch,base, norm=False),ConvD(base,base*2),ConvD(base*2,base*4),ConvD(base*4,base*8,s=1),nn.Conv2d(base*8,1,4,1,1))
    def forward(self,x): return self.net(x)
class MultiScaleDiscriminator(nn.Module):
    def __init__(self,in_ch=2,base=64): super().__init__(); self.d1=PatchDiscriminator(in_ch,base); self.d2=PatchDiscriminator(in_ch,base//2); self.pool=nn.AvgPool2d(3,2,1, count_include_pad=False)
    def forward(self,x): o1=self.d1(x); o2=self.d2(self.pool(x)); return [o1,o2]
