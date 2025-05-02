import torch
import nibabel as nib
import numpy as np
import argparse
from models.unet3d import UNet3D

parser = argparse.ArgumentParser()
parser.add_argument('--t1', type=str, required=True, help='Path to T1 image')
parser.add_argument('--init_seg', type=str, required=True, help='Path to initial segmentation')
parser.add_argument('--output', type=str, required=True, help='Output path for predicted QC segmentation')
parser.add_argument('--weights', type=str, default='model.pth', help='Path to trained model weights')
args = parser.parse_args()

# Load model
model = UNet3D(in_channels=2, out_channels=3).cuda()
model.load_state_dict(torch.load(args.weights))
model.eval()

# Load input
t1 = nib.load(args.t1)
init_seg = nib.load(args.init_seg)

t1_data = t1.get_fdata()
seg_data = init_seg.get_fdata()
x = np.stack([t1_data, seg_data], axis=0)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).cuda()

# Predict
with torch.no_grad():
    pred = model(x)
    pred_class = torch.argmax(pred, dim=1).cpu().squeeze().numpy()

# Save output
pred_nii = nib.Nifti1Image(pred_class.astype(np.uint8), affine=t1.affine)
nib.save(pred_nii, args.output)
