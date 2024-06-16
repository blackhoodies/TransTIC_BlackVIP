
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.visual_prompters import __prompters__
from compressai.models.tic import Alignment

class ModuleWrapper(nn.Module):
    def __init__(self, coordinator_enc, coordinator_dec):
        super().__init__()
        self.coordinator_enc = coordinator_enc
        self.coordinator_dec = coordinator_dec
    def forward(self):
        return 0 
    
class wCoordinator(nn.Module):
    def __init__(self, args, net):
        super().__init__()
        self.net = net
        
        self.p_eps = args.BLACKVIP['P_EPS']
        self.coordinator_enc = __prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Instance') # Task
        self.coordinator_dec = __prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Task')
        
        self.wrapper = ModuleWrapper(__prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Instance').dec,__prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Task').dec)
        self.align = Alignment(64)

    def forward(self, image):
        ins_prompt, _  = self.coordinator_enc(image)
        prompted_images = image + self.p_eps * ins_prompt

        prompted_images = self.align.align(prompted_images)     # [4, 3, 256, 256]
        out = self.net(prompted_images)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        
        recon_image = self.align.resume(out['x_hat'])
        task_prompt, _  = self.coordinator_enc(recon_image)
        out['x_hat'] = recon_image + self.p_eps * task_prompt

        return out
