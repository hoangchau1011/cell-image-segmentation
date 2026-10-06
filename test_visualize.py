import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import BraTSDataset


# ============================================================
# CẤU HÌNH
# ============================================================

IMAGE_DIR = r"D:\BraTS-GLI\processed\images"
MASK_DIR = r"D:\BraTS-GLI\processed\masks"
LIST_FILE = r"D:\BraTS-GLI\processed\val.txt"

OUTPUT_DIR = "visualizations"

NUM_IMAGES = 5


# ============================================================
# Tạo thư mục lưu ảnh
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

dataset = BraTSDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    list_file=LIST_FILE,
    image_size=(256, 256),
    boundary_width=1
)


print()
print("=" * 60)
print("VISUALIZATION")
print("=" * 60)
print("Dataset size:", len(dataset))
print()


# ============================================================
# Chọn một số ảnh
# ============================================================

indices = np.linspace(
    0,
    len(dataset) - 1,
    NUM_IMAGES,
    dtype=int
)


# ============================================================
# Visualization
# ============================================================

for i, idx in enumerate(indices):

    image, mask = dataset[idx]

    # --------------------------------------------------------
    # Tensor → NumPy
    # --------------------------------------------------------

    image_np = image.squeeze(0).numpy()

    mask_np = mask.numpy()

    # --------------------------------------------------------
    # Kiểm tra class
    # --------------------------------------------------------

    unique_values = np.unique(mask_np)

    print(
        f"Image {i + 1}: index = {idx}"
    )

    print(
        "Mask values:",
        unique_values
    )

    # --------------------------------------------------------
    # Tạo ảnh segmentation
    #
    # 0 = background
    # 1 = tumor
    # 2 = boundary
    # --------------------------------------------------------

    segmentation = np.zeros(
        (256, 256, 3),
        dtype=np.uint8
    )

    # Tumor
    segmentation[mask_np == 1] = [
        255,
        0,
        0
    ]

    # Boundary
    segmentation[mask_np == 2] = [
        0,
        255,
        0
    ]

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        (image_np * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    overlay = cv2.addWeighted(
        image_rgb,
        0.7,
        segmentation,
        0.3,
        0
    )

    # ========================================================
    # Vẽ figure
    # ========================================================

    plt.figure(
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original MRI
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image_np,
        cmap="gray"
    )

    plt.title("MRI Image")

    plt.axis("off")

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        mask_np,
        cmap="viridis",
        vmin=0,
        vmax=2
    )

    plt.title(
        "3-Class Mask"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        "Overlay"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Title chung
    # --------------------------------------------------------

    plt.suptitle(
        f"Sample {i + 1} | Dataset index: {idx}"
    )

    plt.tight_layout()

    # ========================================================
    # Lưu ảnh
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{i + 1}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )

    print("-" * 60)


print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(
    f"Images saved to: {OUTPUT_DIR}"
)