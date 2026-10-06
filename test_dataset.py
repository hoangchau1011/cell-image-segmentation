import torch
from dataset import BraTSDataset


IMAGE_DIR = r"D:\BraTS-GLI\processed\images"
MASK_DIR = r"D:\BraTS-GLI\processed\masks"

TRAIN_LIST = r"D:\BraTS-GLI\processed\train.txt"
VAL_LIST = r"D:\BraTS-GLI\processed\val.txt"


# ============================================================
# TRAIN DATASET
# ============================================================

train_dataset = BraTSDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    list_file=TRAIN_LIST
)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = BraTSDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    list_file=VAL_LIST
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)

print(
    "Train dataset:",
    len(train_dataset)
)

print(
    "Validation dataset:",
    len(val_dataset)
)

print("=" * 60)


# Lấy thử 1 sample

image, mask = train_dataset[0]


print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)

print("Image min:", image.min().item())
print("Image max:", image.max().item())

print(
    "Mask values:",
    torch.unique(mask)
)
