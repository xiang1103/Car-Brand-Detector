import torch
import torch.nn.functional as F
import numpy as np

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
