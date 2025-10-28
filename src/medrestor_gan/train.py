
import os, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from models import UNetGenerator, MultiScaleDiscriminator
from dataset import MedRestoreDataset
from losses import d_hinge_loss, g_hinge_loss, ssim, PerceptualLike

def requires_grad(m, flag=True):
    for p in m.parameters(): p.requires_grad=flag

def train(a):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=MedRestoreDataset(a.data, size=a.size)
    dl=DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=2, pin_memory=True)
    G=UNetGenerator(in_ch=2).to(device); D=MultiScaleDiscriminator(in_ch=2).to(device)
    opt_g=torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5,0.999))
    opt_d=torch.optim.Adam(D.parameters(), lr=a.lr*0.5, betas=(0.5,0.999))
    perc=PerceptualLike().to(device)

    os.makedirs(a.out, exist_ok=True)
    step=0
    for epoch in range(a.epochs):
        for b in dl:
            step+=1
            inp=b["input"].to(device); clean=b["clean"].to(device)

            # D
            requires_grad(D, True); requires_grad(G, False)
            with torch.no_grad(): fake=G(inp)
            real_pair=torch.cat([inp[:,0:1], clean],1); fake_pair=torch.cat([inp[:,0:1], fake],1)
            pr, pf = D(real_pair), D(fake_pair)
            loss_d=d_hinge_loss(pr,pf); opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # G
            requires_grad(D, False); requires_grad(G, True)
            fake=G(inp); pf=D(torch.cat([inp[:,0:1], fake],1))
            adv=g_hinge_loss(pf); l1=F.l1_loss(fake,clean); s=1-ssim(fake,clean); pl=perc(fake,clean)
            loss_g=adv + a.l1*l1 + a.perc*pl + a.ssim*s
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            if step % a.logint == 0:
                print(f"[e{epoch}] step {step} | D {loss_d.item():.3f} | G {loss_g.item():.3f} (adv {adv.item():.3f} l1 {l1.item():.3f} ssim {s.item():.3f} perc {pl.item():.3f})")

        torch.save(G.state_dict(), os.path.join(a.out, f"G_epoch{epoch}.pt"))

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/checkpoints")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1", type=float, default=10.0)
    ap.add_argument("--perc", type=float, default=0.2)
    ap.add_argument("--ssim", type=float, default=0.5)
    ap.add_argument("--logint", type=int, default=10)
    a=ap.parse_args()
    train(a)
