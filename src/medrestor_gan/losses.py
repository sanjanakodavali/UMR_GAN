
import torch, torch.nn.functional as F, torch.nn as nn
def d_hinge_loss(real_preds, fake_preds):
    return sum(torch.mean(F.relu(1.-pr)) + torch.mean(F.relu(1.+pf)) for pr,pf in zip(real_preds,fake_preds))
def g_hinge_loss(fake_preds):
    return sum(-torch.mean(pf) for pf in fake_preds)
def _gaussian_window(ch, window_size=11, sigma=1.5, device="cpu"):
    import torch; gauss=torch.Tensor([torch.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to(device); gauss=(gauss/gauss.sum())
    w=(gauss.unsqueeze(1)@gauss.unsqueeze(0)).expand(ch,1,window_size,window_size).contiguous(); return w
def ssim(img1,img2,window_size=11,size_average=True):
    import torch, torch.nn.functional as F
    device=img1.device; ch=img1.size(1); window=_gaussian_window(ch,window_size,1.5,device=device)
    mu1=F.conv2d(img1,window,padding=window_size//2,groups=ch); mu2=F.conv2d(img2,window,padding=window_size//2,groups=ch)
    mu1_sq=mu1.pow(2); mu2_sq=mu2.pow(2); mu1_mu2=mu1*mu2
    sigma1_sq=F.conv2d(img1*img1,window,padding=window_size//2,groups=ch)-mu1_sq
    sigma2_sq=F.conv2d(img2*img2,window,padding=window_size//2,groups=ch)-mu2_sq
    sigma12=F.conv2d(img1*img2,window,padding=window_size//2,groups=ch)-mu1_mu2
    C1=0.01**2; C2=0.03**2
    ssim_map=((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])
class PerceptualLike(nn.Module):
    def __init__(self): super().__init__(); import torch; k=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3); self.register_buffer("lap", k)
    def forward(self,x,y): import torch.nn.functional as F; return F.l1_loss(torch.abs(F.conv2d(x,self.lap,padding=1)), torch.abs(F.conv2d(y,self.lap,padding=1)))
