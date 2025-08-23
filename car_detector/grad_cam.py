import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn 
import torch 
from PIL import Image
from .data_process import denormalize
import torchvision.transforms.functional as TF 

# ---------- Grad-CAM helper ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # forward hook: save activations and attach a grad hook on the tensor
        def fwd_hook(m, inp, out):
            self.activations = out.detach()
            # attach a hook to the tensor to capture its gradient safely
            def grad_hook(grad):
                self.gradients = grad.detach().clone()
            out.register_hook(grad_hook)

        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)

    def __del__(self):
        try:
            self.fwd_handle.remove()
        except Exception:
            pass

    @torch.no_grad()
    def _normalize_cam(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, x, class_idx=None):
        logits = self.model(x)                 # forward populates activations and sets up grad hook
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # backprop only the selected logit(s)
        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(logits.size(0)), class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        acts = self.activations               # (B,K,H',W')
        grads = self.gradients                # (B,K,H',W')
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)     # (B,H',W')
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = torch.stack([self._normalize_cam(c) for c in cam], dim=0)
        return cam, class_idx


def overlay_heatmap(img_pil, cam_tensor, alpha=0.35):
    """cam_tensor: (H,W) in [0,1]
    used to create the activation layer heat map 
    """

    # assume img_pil is a PIL Image 
    import cv2
    cam_np = cam_tensor.cpu().numpy()
    cam_u8 = (cam_np * 255).astype(np.uint8)

    base = img_pil.convert("RGB")
    size = (base.width, base.height)

    hm = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB

    # Blend
    hm = Image.fromarray(hm).resize(size, Image.BILINEAR)  # or Image.Resampling.BILINEAR on newer Pillow
    return Image.blend(base, hm, alpha) 



def visual_heatmap(input_img:torch.Tensor, model,target_layer):
    # reset inplace to avoid torch autograd view error 
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False 

    cam_extractor= GradCAM(model=model, target_layer=target_layer)

    # check if input is normalized 
    if not (input_img.min() >= 0) and (input_img.max() <= 1):
        print("-----normalizing image tensor------")
        img= denormalize(input_img)
    
    # get batch dim 
    img= img.unsqueeze(0) 
    
    # complete activation map extract 
    with torch.enable_grad():  # we need grad, so don't use torch.no_grad
        cam, pred_idx = cam_extractor(img, class_idx=None)

    # change into PIL image 
    img = img.squeeze(0) 

    pil_img = TF.to_pil_image(img)

    blended = overlay_heatmap(pil_img, cam[0])
    return pil_img, blended, pred_idx 