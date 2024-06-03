
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.visual_prompters import __prompters__
from compressai.models.tic import Alignment

from loguru import logger
    
class wCoordinator(nn.Module):
    def __init__(self, args, net):
        super().__init__()
        self.net = net
        
        self.p_eps = args.BLACKVIP['P_EPS']
        self.coordinator_enc = __prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Instance') # Task
        self.coordinator_dec = __prompters__[args.BLACKVIP['METHOD']](args, prompt_type='Task')

        self.align = Alignment(64)

    def forward(self, image):
        ins_prompt, _  = self.coordinator_enc(image)
        prompted_images = image + self.p_eps * ins_prompt

        aligned_prompted_images = self.align.align(prompted_images)     # [4, 3, 256, 256]
        out = self.net(aligned_prompted_images)
        
        recon_image = self.align.resume(out['x_hat'])
        task_prompt, _  = self.coordinator_enc(recon_image)
        out['x_hat'] = recon_image + self.p_eps * task_prompt

        # logger.debug(f"{recon_image.shape=}", f"{ins_prompt.shape=}", f"{task_prompt.shape=}")

        out['prompted_images'] = prompted_images
        out['recon_image'] = recon_image
        out['ins_prompt'] = ins_prompt
        out['task_prompt'] = task_prompt

        return out
