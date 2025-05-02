import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import time


# MONAI Augmentations
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.networks.utils import one_hot

# Set deterministic training for reproducibility
set_determinism(seed=42)

# Configuration
data_dir = "/datasets/tdt4265/mic/open/HNTS-MRG/train" 
model_dir = "./model"
train_batch_size = 1
val_batch_size = 1
num_workers = 4
learning_rate = 1e-4
max_epochs = 300 
val_interval = 1
roi_size = (96, 96, 96)
os.makedirs(model_dir, exist_ok=True)

# Function to collect patient data
def get_data_dicts(data_dir, task="preRT"):
    """
    Create data dictionaries for training and validation
    """
    all_patients = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    train_patients = all_patients[:int(0.8 * len(all_patients))]
    val_patients = all_patients[int(0.8 * len(all_patients)):]
    
    train_files = []
    val_files = []
    
    for patient in train_patients:
        if task == "preRT":
            img_path = os.path.join(data_dir, patient, "preRT", f"{patient}_preRT_T2.nii.gz")
            mask_path = os.path.join(data_dir, patient, "preRT", f"{patient}_preRT_mask.nii.gz")
        else:  # midRT
            img_path = os.path.join(data_dir, patient, "midRT", f"{patient}_midRT_T2.nii.gz")
            mask_path = os.path.join(data_dir, patient, "midRT", f"{patient}_midRT_mask.nii.gz")
        
        # Only add if both files exist
        if os.path.exists(img_path) and os.path.exists(mask_path):
            train_files.append({"image": img_path, "label": mask_path})
    
    for patient in val_patients:
        if task == "preRT":
            img_path = os.path.join(data_dir, patient, "preRT", f"{patient}_preRT_T2.nii.gz")
            mask_path = os.path.join(data_dir, patient, "preRT", f"{patient}_preRT_mask.nii.gz")
        else:  # midRT
            img_path = os.path.join(data_dir, patient, "midRT", f"{patient}_midRT_T2.nii.gz")
            mask_path = os.path.join(data_dir, patient, "midRT", f"{patient}_midRT_mask.nii.gz")
        
        # Only add if both files exist
        if os.path.exists(img_path) and os.path.exists(mask_path):
            val_files.append({"image": img_path, "label": mask_path})
    
    return train_files, val_files

# Define transforms
def get_transforms(mode="train"):
    """
    Returns transforms for training and validation
    """
    # Common transforms for both training and validation
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ]
    
    if mode == "train":
        train_transforms = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + train_transforms)
    else:  # validation transforms
        val_transforms = [
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(common_transforms + val_transforms)

# Get data
train_files, val_files = get_data_dicts(data_dir, task="preRT")
train_transforms = get_transforms(mode="train")
val_transforms = get_transforms(mode="val")

# Create datasets
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=num_workers
)
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=num_workers
)

# Create data loaders
train_loader = DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=list_data_collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=val_batch_size,
    num_workers=num_workers,
    collate_fn=list_data_collate,
)

# Create model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple UNet model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,  # Background (0), GTVp (1), GTVn (2)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training function
def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in loader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Step {step}/{len(loader)}, Loss: {loss.item():.4f}", end="\r")
    epoch_loss /= step
    return epoch_loss

# Validation function
def validate(model, loader, dice_metric, device):
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].long().to(device)

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

            # one-hot encoden (shape: [B, C, ...])
            preds_onehot = one_hot(preds, num_classes=3)
            labels_onehot = one_hot(labels.unsqueeze(1), num_classes=3)

            dice_metric(y_pred=preds_onehot, y=labels_onehot)

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
    return metric

# Main training loop
def train(max_epochs=200):
    best_metric = -1
    best_metric_epoch = -1
    dice_scores = []
    loss_values = []

    start_time = time.time()

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        loss_values.append(epoch_loss)

        if (epoch + 1) % val_interval == 0:
            metric = validate(model, val_loader, dice_metric, device)
            dice_scores.append((epoch + 1, metric))
            print(f"Validation Dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                print(f"New best model saved! Dice: {best_metric:.4f}")

    print(f"Training completed. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Zeit und Energieverbrauch berechnen
    total_time = time.time() - start_time
    total_hours = total_time / 3600.0
    power_draw_watts = 225
    energy_kwh = (power_draw_watts * total_hours) / 1000.0

    # Energiebericht speichern
    with open(os.path.join(model_dir, "energy_usage.txt"), "w") as f:
        f.write(f"Training time: {total_hours:.2f} hours\n")
        f.write(f"Estimated energy usage: {energy_kwh:.2f} kWh\n")
        f.write(f"That's enough to drive a Tesla Model Y approx. {energy_kwh * 6.5:.1f} km (6.5 km/kWh)\n")

    # === Plot 1: Loss
    if dice_scores:
        epochs_dice, dice_vals = zip(*dice_scores)
        epochs_loss = list(range(1, len(loss_values) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1 – Training Loss
        ax1.plot(epochs_loss, loss_values, color='blue', marker='x')
        ax1.set_title("Training Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Subplot 2 – Validation Dice
        ax2.plot(epochs_dice, dice_vals, color='red', marker='o')
        ax2.set_title("Validation Dice over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice Score")
        ax2.grid(True)

        plt.suptitle("Training Progress", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(model_dir, "loss_and_dice_side_by_side.png"))
        plt.close()

    with open(os.path.join(model_dir, "loss_values.json"), "w") as f:
        json.dump(loss_values, f)

    return model

# Run training
if __name__ == "__main__":
    trained_model = train(max_epochs=max_epochs)
