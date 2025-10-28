
import glob, os, random, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
def pil_to_tensor_gray(im): import numpy as np, torch; arr=np.array(im.convert("L"),dtype=np.float32)/255.0; return torch.from_numpy(arr).unsqueeze(0)
def tensor_to_pil_gray(t): import numpy as np; arr=(t.squeeze().clamp(0,1).cpu().numpy()*255).astype(np.uint8); from PIL import Image; return Image.fromarray(arr,'L')
def random_mask(h,w):
    m=np.ones((h,w),dtype=np.float32); 
    for _ in range(np.random.randint(1,4)):
        rh,rw=np.random.randint(h//8,h//3),np.random.randint(w//8,w//3); y=np.random.randint(0,h-rh); x=np.random.randint(0,w-rw); m[y:y+rh,x:x+rw]=0.0
    return m
def add_noise(t):
    sigma=np.random.uniform(0.01,0.08); out=(t + torch.randn_like(t)*sigma).clamp(0,1)
    if random.random()<0.3:
        from PIL import ImageFilter; out = pil_to_tensor_gray(tensor_to_pil_gray(out).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5,1.5))))
    return out
class MedRestoreDataset(Dataset):
    def __init__(self, img_dir, size=256):
        self.paths=sorted(glob.glob(os.path.join(img_dir,"**","*.jpg"), recursive=True)+glob.glob(os.path.join(img_dir,"**","*.png"), recursive=True))
        self.size=size
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]; im=Image.open(p).convert("L").resize((self.size,self.size), Image.BICUBIC)
        clean=pil_to_tensor_gray(im); noisy=add_noise(clean); import torch; m=torch.from_numpy(random_mask(self.size,self.size)).unsqueeze(0); inp=torch.cat([noisy*m, m],0)
        return {"input": inp, "clean": clean, "mask": m, "path": p}
