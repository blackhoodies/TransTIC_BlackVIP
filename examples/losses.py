import torch
import torch.nn as nn
import math
from torchvision.models import resnet50
from torchvision import transforms


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output

class Clsloss(nn.Module):
    def __init__(self, device, perceptual_loss=False) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.classifier = resnet50(True)
        self.classifier.requires_grad_(False)
        self.hooks = [FeatureHook(i) for i in [
            self.classifier.layer1,
            self.classifier.layer2,
            self.classifier.layer3,
            self.classifier.layer4,
        ]]
        self.classifier = self.classifier.to(device)
        for k, p in self.classifier.named_parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, output, d, y_true):
        x_hat = torch.clamp(output["x_hat"],0,1)
        pred = self.classifier(self.transform(x_hat))
        loss = self.ce(pred, y_true)
        accu = sum(torch.argmax(pred,-1)==y_true)/pred.shape[0]
        if self.perceptual_loss:
            pred_feat = [i.feature.clone() for i in self.hooks]
            _ = self.classifier(self.transform(d))
            ori_feat = [i.feature.clone() for i in self.hooks]
            perc_loss = torch.stack([nn.functional.mse_loss(p,o, reduction='none').mean((1,2,3)) for p,o in zip(pred_feat, ori_feat)])
            perc_loss = perc_loss.mean()
            return loss, accu, perc_loss

        return loss, accu, None

class Loss():
    def __init__(self, model, device, perceptual_loss=False, lmbda=1e-2):
        super().__init__()
        self.model = model
        self.rd_loss = RateDistortionLoss(lmbda=lmbda)
        self.cls_loss = Clsloss(device=device, perceptual_loss=perceptual_loss)
    
    def loss_fn(self, output, data, label, lmbda):
        out_criterion = self.rd_loss(output, data)
        loss, accu, perc_loss = self.cls_loss(output, data, label)
        total_loss = 1000*lmbda*perc_loss + out_criterion['bpp_loss']
        return total_loss, out_criterion, loss, accu, perc_loss
        
    def spsa_grad_estimate_bi(self, w, model, data, label, lmbda, ck):
        #* repeat k times and average them for stabilizing
        ghats = []
        N_params = len(torch.nn.utils.parameters_to_vector(model.wrapper.parameters()))
        # N_params = len(torch.nn.utils.parameters_to_vector(model.coordinator_dec.dec.parameters()))
        
        sp_avg = 5
        for spk in range(sp_avg):
            #! Bernoulli {-1, 1}
            # perturb = torch.bernoulli(torch.empty(self.N_params).uniform_(0,1)).cuda()
            # perturb[perturb < 1] = -1
            #! Segmented Uniform [-1, 0.5] U [0.5, 1]
            p_side = (torch.rand(N_params).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side

            #* two-side Approximated Numerical Gradient
            w_r = w + ck*perturb
            w_l = w - ck*perturb
            # if prompt_type == 'instance':
            #     torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator_enc.dec.parameters())
            #     output1 = model(data)
            #     torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator_enc.dec.parameters())
            #     output2 = model(data)
            # else:
        
            torch.nn.utils.vector_to_parameters(w_r, model.wrapper.parameters())
            w_wrapper_ed = torch.nn.utils.parameters_to_vector(model.wrapper.coordinator_enc.parameters())
            torch.nn.utils.vector_to_parameters(w_wrapper_ed, model.coordinator_enc.dec.parameters())
            w_wrapper_dd = torch.nn.utils.parameters_to_vector(model.wrapper.coordinator_dec.parameters())
            torch.nn.utils.vector_to_parameters(w_wrapper_dd, model.coordinator_dec.dec.parameters())
            # model.coordinator_enc.dec.weight = model.wrapper.coordinator_enc.dec.weight.clone()
            # model.coordinator_dec.dec.weight = model.wrapper.coordinator_dec.dec.weight.clone()
            output1 = model(data)
            torch.nn.utils.vector_to_parameters(w_l, self.model.wrapper.parameters())
            w_wrapper_ed = torch.nn.utils.parameters_to_vector(model.wrapper.coordinator_enc.parameters())
            torch.nn.utils.vector_to_parameters(w_wrapper_ed, model.coordinator_enc.dec.parameters())
            w_wrapper_dd = torch.nn.utils.parameters_to_vector(model.wrapper.coordinator_dec.parameters())
            torch.nn.utils.vector_to_parameters(w_wrapper_dd, model.coordinator_dec.dec.parameters())
            # model.coordinator_enc.dec.weight = model.wrapper.coordinator_enc.dec.weight.clone()
            # model.coordinator_dec.dec.weight = model.wrapper.coordinator_dec.dec.weight.clone()
            output2 = model(data)
     
            # torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator_enc.dec.parameters())
            # torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator_enc.dec.parameters())
            # torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator_dec.dec.parameters())
            # torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator_dec.dec.parameters())
            total_loss1, out_criterion1, loss1, accu1, perc_loss1 = self.loss_fn(output1, data, label, lmbda)
            total_loss2, out_criterion2, loss2, accu2, perc_loss2 = self.loss_fn(output2, data, label, lmbda)

            #* parameter update via estimated gradient
            ghat = (total_loss1 - total_loss2)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
          
        if sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0) 
        total_loss = ((total_loss1 + total_loss2)/2)
        acc = (0.5*(accu1 + accu2)).item()
        return ghat, total_loss, acc, out_criterion1, perc_loss1, loss1, model
    

    
def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res