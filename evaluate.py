import os
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from dataset import NucleiDataset
from unet_model import UNet



# ======================================================
# CONFIG
# ======================================================

IMAGE_DIR = "data/nuclei_processed/images"
MASK_DIR = "data/nuclei_processed/masks"

MODEL_PATH = "checkpoints/best_model.pth"

SAVE_DIR = "prediction_results"

NUM_CLASSES = 3

BATCH_SIZE = 8

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



# ======================================================
# DICE SCORE
# ======================================================

def dice_score(outputs, masks, num_classes=3):

    preds = torch.argmax(
        outputs,
        dim=1
    )

    dices = []

    for cls in range(num_classes):

        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()

        intersection = (
            pred_cls *
            mask_cls
        ).sum()

        denominator = (
            pred_cls.sum()
            +
            mask_cls.sum()
        )

        if denominator == 0:

            dices.append(
                torch.tensor(
                    1.0,
                    device=outputs.device
                )
            )

        else:

            dices.append(
                (2 * intersection + 1e-7)
                /
                (denominator + 1e-7)
            )

    return torch.mean(
        torch.stack(dices)
    )



# ======================================================
# MEAN IOU
# ======================================================

def mean_iou(outputs, masks, num_classes=3):

    preds = torch.argmax(
        outputs,
        dim=1
    )

    ious = []

    for cls in range(num_classes):

        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (
            pred_cls &
            mask_cls
        ).sum().item()

        union = (
            pred_cls |
            mask_cls
        ).sum().item()

        if union == 0:

            ious.append(1.0)

        else:

            ious.append(
                intersection / union
            )

    return np.mean(ious)



# ======================================================
# PER-CLASS DICE
# ======================================================

def dice_per_class(outputs, masks, num_classes=3):

    preds = torch.argmax(
        outputs,
        dim=1
    )

    result = []

    for cls in range(num_classes):

        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()

        intersection = (
            pred_cls *
            mask_cls
        ).sum()

        denominator = (
            pred_cls.sum()
            +
            mask_cls.sum()
        )

        if denominator == 0:

            result.append(1.0)

        else:

            result.append(

                (
                    (2 * intersection + 1e-7)
                    /
                    (denominator + 1e-7)
                ).item()

            )

    return result



# ======================================================
# CREATE OVERLAY
# ======================================================

def create_overlay(image, prediction):

    image = (
        image.permute(1,2,0).numpy() * 255
    ).astype(np.uint8)

    overlay = image.copy()

    colors = {

        1:(0,255,0),      # nucleus

        2:(255,0,0)       # boundary

    }

    for cls,color in colors.items():

        overlay[
            prediction == cls
        ] = color

    overlay = cv2.addWeighted(

        image,

        0.6,

        overlay,

        0.4,

        0

    )

    return overlay



# ======================================================
# SAVE RESULT
# ======================================================

def save_prediction(image,
                    mask,
                    prediction,
                    save_path):

    overlay = create_overlay(
        image,
        prediction
    )

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.imshow(image.permute(1,2,0))
    plt.title("Image")
    plt.axis("off")


    plt.subplot(1,4,2)
    plt.imshow(
        mask,
        cmap="viridis"
    )
    plt.title("Ground Truth")
    plt.axis("off")


    plt.subplot(1,4,3)
    plt.imshow(
        prediction,
        cmap="viridis"
    )
    plt.title("Prediction")
    plt.axis("off")


    plt.subplot(1,4,4)
    plt.imshow(
        overlay
    )
    plt.title("Overlay")
    plt.axis("off")


    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    # ======================================================
# MAIN
# ======================================================

def main():

    print("Device:", DEVICE)

    # ----------------------------
    # DATASET
    # ----------------------------

    dataset = NucleiDataset(
        IMAGE_DIR,
        MASK_DIR,
        train=False
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print("Validation:", len(val_dataset))

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ----------------------------
    # MODEL
    # ----------------------------

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    print("Loaded model:", MODEL_PATH)

    # ----------------------------
    # METRICS
    # ----------------------------

    total_dice = 0
    total_iou = 0

    class_dice = np.zeros(NUM_CLASSES)
    class_iou = np.zeros(NUM_CLASSES)

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            total_dice += dice_score(
                outputs,
                masks,
                NUM_CLASSES
            ).item()

            total_iou += mean_iou(
                outputs,
                masks,
                NUM_CLASSES
            )

            class_dice += np.array(
                dice_per_class(
                    outputs,
                    masks,
                    NUM_CLASSES
                )
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            for cls in range(NUM_CLASSES):

                pred_cls = preds == cls
                mask_cls = masks == cls

                intersection = (
                    pred_cls &
                    mask_cls
                ).sum().item()

                union = (
                    pred_cls |
                    mask_cls
                ).sum().item()

                if union == 0:
                    class_iou[cls] += 1
                else:
                    class_iou[cls] += intersection / union

    total_dice /= len(loader)
    total_iou /= len(loader)

    class_dice /= len(loader)
    class_iou /= len(loader)

    # ----------------------------
    # PRINT RESULT
    # ----------------------------

    print("\n==============================")
    print(f"Overall Dice : {total_dice:.4f}")
    print(f"Overall IoU  : {total_iou:.4f}")
    print("==============================")

    class_names = [
        "Background",
        "Nucleus",
        "Boundary"
    ]

    print("\nPer-class Performance")
    print("-" * 40)

    for i in range(NUM_CLASSES):

        print(
            f"{class_names[i]:12s}"
            f" Dice = {class_dice[i]:.4f}"
            f" | IoU = {class_iou[i]:.4f}"
        )

    # ----------------------------
    # SAVE 10 PREDICTIONS
    # ----------------------------

    indices = random.sample(
        range(len(val_dataset)),
        min(10, len(val_dataset))
    )

    for i, idx in enumerate(indices):

        image, mask = val_dataset[idx]

        image_input = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(image_input)

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()

        save_prediction(

            image,

            mask.numpy(),

            prediction,

            os.path.join(
                SAVE_DIR,
                f"prediction_{i+1:03d}.png"
            )

        )

    print(
        f"\nSaved {len(indices)} prediction images"
    )

    print(
        f"Folder : {SAVE_DIR}"
    )


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    main()