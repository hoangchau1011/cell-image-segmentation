import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# LOAD HISTORY
# ======================================================

history = np.load(
    "checkpoints/history.npy",
    allow_pickle=True
).item()


train_loss = history["train_loss"]
val_loss = history["val_loss"]
dice = history["dice"]
iou = history["iou"]


epochs = range(
    1,
    len(train_loss) + 1
)


# ======================================================
# LOSS
# ======================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ======================================================
# DICE
# ======================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    dice,
    label="Dice Score",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Dice Score")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "dice_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ======================================================
# IoU
# ======================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    iou,
    label="Mean IoU",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Mean IoU")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "iou_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ======================================================
# SUMMARY
# ======================================================

print("=" * 40)

print(f"Best Dice     : {max(dice):.4f}")
print(f"Best IoU      : {max(iou):.4f}")

print(f"Lowest Train Loss : {min(train_loss):.4f}")
print(f"Lowest Val Loss   : {min(val_loss):.4f}")

print("=" * 40)