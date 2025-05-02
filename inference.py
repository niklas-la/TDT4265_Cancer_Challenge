import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensity, CropForeground, ResizeWithPadOrCrop, ToTensor
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import os

# Configuration
model_path = "model/best_model.pth"
input_image_path = "49_preRT_T2.nii.gz"
output_mask_path = "prediction_mask.nii.gz"
roi_size = (96, 96, 96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3

# Modle
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing transform
transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensity(),
    CropForeground(),
    ResizeWithPadOrCrop(spatial_size=roi_size),
    ToTensor()
])

# Load image
img_tensor = transform(input_image_path).unsqueeze(0).to(device)  # shape [1, 1, D, H, W]

# Inference
with torch.no_grad():
    output = sliding_window_inference(img_tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)  # shape [1, D, H, W]

pred_np = pred[0].cpu().numpy().astype(np.uint8)

original_img = nib.load(input_image_path)
affine = original_img.affine

nib.save(nib.Nifti1Image(pred_np, affine), output_mask_path)
print(f"Saved prediction to: {output_mask_path}")
