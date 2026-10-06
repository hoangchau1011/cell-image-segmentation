import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import NucleiDataset
from unet_model import UNet


# ======================================================
# CONFIG
# ======================================================

IMAGE_DIR = "data/nuclei_processed/images"
MASK_DIR = "data/nuclei_processed/masks"

SAVE_DIR = "checkpoints"

NUM_CLASSES = 3

BATCH_SIZE = 8

EPOCHS = 300

LEARNING_RATE = 1e-4

SEED = 42


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ======================================================
# FIX RANDOM
# ======================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ======================================================
# METRICS
# ======================================================

def dice_score(
        outputs,
        masks,
        num_classes=3
):

    preds = torch.argmax(
        outputs,
        dim=1
    )

    dice = []

    for cls in range(num_classes):

        pred_cls = (
            preds == cls
        ).float()

        mask_cls = (
            masks == cls
        ).float()

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

            dice.append(
                torch.tensor(
                    1.0,
                    device=outputs.device
                )
            )

        else:

            dice.append(
                (2 * intersection + 1e-7)
                /
                (denominator + 1e-7)
            )

    return torch.mean(
        torch.stack(dice)
    )


def mean_iou(
        outputs,
        masks,
        num_classes=3
):

    preds = torch.argmax(
        outputs,
        dim=1
    )

    ious = []

    for cls in range(num_classes):

        pred_cls = (
            preds == cls
        )

        mask_cls = (
            masks == cls
        )

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
                intersection /
                union
            )

    return np.mean(
        ious
    )


# ======================================================
# TRAIN ONE EPOCH
# ======================================================

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        scaler
):

    model.train()

    total_loss = 0

    bar = tqdm(
        loader,
        desc="Training"
    )

    for images, masks in bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

        bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader)


# ======================================================
# VALIDATION
# ======================================================

def validate(
        model,
        loader,
        criterion
):

    model.eval()

    total_loss = 0

    total_dice = 0

    total_iou = 0

    with torch.no_grad():

        bar = tqdm(
            loader,
            desc="Validation"
        )

        for images, masks in bar:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            total_loss += loss.item()

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

    return (

        total_loss / len(loader),

        total_dice / len(loader),

        total_iou / len(loader)

    )
# ======================================================
# PLOT TRAINING HISTORY
# ======================================================

def plot_history(history):

    # Loss
    plt.figure(figsize=(6,4))

    plt.plot(
        history["train_loss"],
        label="Train"
    )

    plt.plot(
        history["val_loss"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "loss_curve.png"
        ),
        dpi=300
    )

    plt.close()


    # Dice

    plt.figure(figsize=(6,4))

    plt.plot(
        history["dice"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Score")
    plt.grid(True)

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "dice_curve.png"
        ),
        dpi=300
    )

    plt.close()


    # IoU

    plt.figure(figsize=(6,4))

    plt.plot(
        history["iou"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Mean IoU")
    plt.grid(True)

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "iou_curve.png"
        ),
        dpi=300
    )

    plt.close()


# ======================================================
# SAVE PREDICTIONS
# ======================================================

def save_predictions(
        model,
        dataset,
        num_images=20
):

    prediction_dir = os.path.join(
        SAVE_DIR,
        "prediction_results"
    )

    os.makedirs(
        prediction_dir,
        exist_ok=True
    )


    model.eval()


    with torch.no_grad():

        for i in range(
            min(
                num_images,
                len(dataset)
            )
        ):

            image, mask = dataset[i]

            output = model(

                image.unsqueeze(0).to(
                    DEVICE
                )

            )


            pred = torch.argmax(
                output,
                dim=1
            )[0].cpu().numpy()


            fig, ax = plt.subplots(
                1,
                3,
                figsize=(12,4)
            )


            ax[0].imshow(

                image.permute(
                    1,
                    2,
                    0
                )

            )

            ax[0].set_title(
                "Image"
            )

            ax[0].axis("off")


            ax[1].imshow(
                mask.numpy(),
                cmap="viridis"
            )

            ax[1].set_title(
                "Ground Truth"
            )

            ax[1].axis("off")


            ax[2].imshow(
                pred,
                cmap="viridis"
            )

            ax[2].set_title(
                "Prediction"
            )

            ax[2].axis("off")


            plt.tight_layout()


            plt.savefig(

                os.path.join(

                    prediction_dir,

                    f"prediction_{i+1:03d}.png"

                ),

                dpi=300,

                bbox_inches="tight"

            )


            plt.close()


    print()

    print("=========================")
    print("Prediction images saved!")
    print("Folder:", prediction_dir)
    print("=========================")
    # ======================================================
# MAIN
# ======================================================

def main():

    print("Device:", DEVICE)

    # ==========================
    # DATASET
    # ==========================

    train_full = NucleiDataset(
        IMAGE_DIR,
        MASK_DIR,
        train=True
    )

    val_full = NucleiDataset(
        IMAGE_DIR,
        MASK_DIR,
        train=False
    )

    train_size = int(
        0.8 * len(train_full)
    )

    val_size = len(train_full) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, _ = random_split(
        train_full,
        [train_size, val_size],
        generator=generator
    )

    _, val_dataset = random_split(
        val_full,
        [train_size, val_size],
        generator=generator
    )

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ==========================
    # MODEL
    # ==========================

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "dice": [],
        "iou": []
    }

    best_dice = 0

    # ==========================
    # TRAIN LOOP
    # ==========================

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        val_loss, dice, iou = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(dice)

        print(f"""
Train Loss : {train_loss:.4f}
Val Loss   : {val_loss:.4f}
Dice       : {dice:.4f}
IoU        : {iou:.4f}
""")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dice"].append(dice)
        history["iou"].append(iou)

        # Save last model
        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_DIR,
                "last_model.pth"
            )
        )

        # Save best model
        if dice > best_dice:

            best_dice = dice

            torch.save(
                model.state_dict(),
                os.path.join(
                    SAVE_DIR,
                    "best_model.pth"
                )
            )

            print("⭐ Saved Best Model!")

    # ==========================
    # SAVE HISTORY
    # ==========================

    np.save(
        os.path.join(
            SAVE_DIR,
            "history.npy"
        ),
        history
    )

    # ==========================
    # PLOT HISTORY
    # ==========================

    plot_history(history)

    # ==========================
    # LOAD BEST MODEL
    # ==========================

    best_model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    best_model.load_state_dict(

        torch.load(

            os.path.join(
                SAVE_DIR,
                "best_model.pth"
            ),

            map_location=DEVICE

        )

    )

    # ==========================
    # SAVE PREDICTION IMAGES
    # ==========================

    save_predictions(

        best_model,

        val_dataset,

        num_images=20

    )

    print("\n============================")
    print("Training Finished!")
    print(f"Best Dice : {best_dice:.4f}")
    print("Results saved in:", SAVE_DIR)
    print("============================")


if __name__ == "__main__":

    main()