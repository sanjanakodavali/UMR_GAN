
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay; self.shadow={k:v.clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items(): self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        for k,v in model.state_dict().items(): v.copy_(self.shadow[k])
