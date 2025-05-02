import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityd, CropForegroundd,
    ResizeWithPadOrCropd, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from glob import glob

# Configuration
test_dir = "/datasets/tdt4265/mic/open/HNTS-MRG/test"
model_path = "model/best_model.pth"
roi_size = (96, 96, 96)
batch_size = 1
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_files(test_dir):
    test_files = []
    patients = sorted(os.listdir(test_dir))
    for patient in patients:
        subdir = os.path.join(test_dir, patient, "preRT")
        image_path = os.path.join(subdir, f"{patient}_preRT_T2.nii.gz")
        label_path = os.path.join(subdir, f"{patient}_preRT_mask.nii.gz")
        if os.path.exists(image_path) and os.path.exists(label_path):
            test_files.append({"image": image_path, "label": label_path})
        else:
            print(f"Missing data for patient {patient}, skipping.")
    return test_files

# Dice Calculation per class (manual)
def dice_score_per_class(preds, labels, num_classes=3):
    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        label_c = (labels == c).float()
        intersection = (pred_c * label_c).sum()
        denominator = pred_c.sum() + label_c.sum()
        if denominator == 0:
            dice = torch.tensor(1.0)  # Perfect match if both empty
        else:
            dice = (2. * intersection) / denominator
        dice_scores.append(dice.item())
    return dice_scores

# Transforms
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size),
    ToTensord(keys=["image", "label"]),
])

# Load test data
test_files = get_test_files(test_dir)
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

# Load model
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

# Evaluation
all_dice = []

with torch.no_grad():
    for test_data in test_loader:
        inputs = test_data["image"].to(device)
        labels = test_data["label"].long().to(device)

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size=1, predictor=model)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        dice_scores = dice_score_per_class(preds[0], labels[0], num_classes=num_classes)
        all_dice.append(dice_scores)

# Aggregate results
all_dice_np = np.array(all_dice)  # [N, 3]
mean_dice = all_dice_np.mean(axis=0)

# Report
print("Dice Score per Class (including when not present):")
print(f"Background (class 0): {mean_dice[0]:.4f}")
print(f"GTVp       (class 1): {mean_dice[1]:.4f}")
print(f"GTVn       (class 2): {mean_dice[2]:.4f}")
print(f"Mean Dice (GTVp + GTVn): {(mean_dice[1:3].mean()):.4f}")
