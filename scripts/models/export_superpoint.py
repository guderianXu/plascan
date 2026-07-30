"""Export SuperPoint for PlaScan C++ extractor interface.
C++ expects: forward(image [1,1,H,W], orig_wh [W,H]) -> (kpts [N,2], scores [N], desc_dense [1,256,H/8,W/8])
"""
import os
import torch, torch.nn as nn, torch.nn.functional as F
from lightglue import SuperPoint
from pathlib import Path

OUT = Path(os.environ.get(
    "PLASCAN_MODEL_OUT",
    Path(__file__).resolve().parents[2] / "resources" / "models",
))

class SuperPointExtractor(nn.Module):
    def __init__(self, max_kpts=2048):
        super().__init__()
        self.sp = SuperPoint(max_num_keypoints=max_kpts).eval()

    def forward(self, image, orig_wh):
        # image: [1,1,H,W] grayscale, sp.conv1a expects 1 channel
        x = self.sp.relu(self.sp.conv1a(image))
        x = self.sp.relu(self.sp.conv1b(x))
        x = self.sp.pool(x)
        x = self.sp.relu(self.sp.conv2a(x))
        x = self.sp.relu(self.sp.conv2b(x))
        x = self.sp.pool(x)
        x = self.sp.relu(self.sp.conv3a(x))
        x = self.sp.relu(self.sp.conv3b(x))
        x = self.sp.pool(x)
        x = self.sp.relu(self.sp.conv4a(x))
        x = self.sp.relu(self.sp.conv4b(x))
        # Score map
        cPa = self.sp.relu(self.sp.convPa(x))
        sc = F.softmax(self.sp.convPb(cPa), 1)[:, :-1]
        b, _, h, w = sc.shape
        sc = sc.permute(0,2,3,1).reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
        # Dense descriptors [1,256,H/8,W/8]
        desc_dense = self.sp.convDb(self.sp.relu(self.sp.convDa(x)))
        # Top-k keypoints
        flat = sc.reshape(b, -1)
        k = min(2048, flat.shape[1])
        vals, idx = torch.topk(flat, k, dim=1)
        Hout, Wout = h * 8, w * 8
        kpts = torch.stack([(idx % Wout).float(), (idx // Wout).float()], dim=-1)  # [1,N,2]
        return kpts.squeeze(0), vals.squeeze(0), desc_dense  # [N,2],[N],[1,256,H/8,W/8]

print("Exporting SuperPoint extractor...")
OUT.mkdir(parents=True, exist_ok=True)
model = SuperPointExtractor(max_kpts=2048)
traced = torch.jit.trace(model, (torch.rand(1,1,480,640), torch.tensor([640.,480.])), strict=False)
path = OUT / "superpoint_extractor_cpu.torchscript"
traced.save(str(path))
print(f"OK: {path}")
