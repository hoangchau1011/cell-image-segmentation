import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================
# ĐƯỜNG DẪN
# ==========================

# Thư mục chứa dataset sau khi download
SOURCE_DIR = "./data/archive/stage1_train"

# Thư mục lưu dataset đã chuẩn hóa
OUTPUT_IMAGE_DIR = "./data/nuclei_processed/images"
OUTPUT_MASK_DIR = "./data/nuclei_processed/masks"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# ==========================
# KIỂM TRA DATASET
# ==========================

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dataset:\n{SOURCE_DIR}\n"
        "Hãy chạy download_dataset.py trước."
    )

sample_folders = sorted(
    [
        folder
        for folder in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, folder))
    ]
)

print(f"Tìm thấy {len(sample_folders)} mẫu.")

# ==========================
# XỬ LÝ DATASET
# ==========================

for idx, sample_id in enumerate(tqdm(sample_folders)):

    sample_path = os.path.join(SOURCE_DIR, sample_id)

    image_folder = os.path.join(sample_path, "images")
    mask_folder = os.path.join(sample_path, "masks")

    # --------------------------
    # Đọc ảnh gốc
    # --------------------------

    image_name = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    # --------------------------
    # Gộp tất cả mask
    # --------------------------

    combined_mask = None

    mask_files = sorted(os.listdir(mask_folder))

    for mask_file in mask_files:

        mask_path = os.path.join(mask_folder, mask_file)

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if combined_mask is None:
            combined_mask = np.zeros_like(mask)

        combined_mask = np.maximum(combined_mask, mask)

    # Chuyển thành mask nhị phân
    combined_mask = (combined_mask > 0).astype(np.uint8) * 255

    # --------------------------
    # Lưu ảnh và mask
    # --------------------------

    save_name = f"{idx:04d}.png"

    cv2.imwrite(
        os.path.join(OUTPUT_IMAGE_DIR, save_name),
        image
    )

    cv2.imwrite(
        os.path.join(OUTPUT_MASK_DIR, save_name),
        combined_mask
    )

print("\nHoàn tất chuẩn hóa dataset!")
print("Ảnh:", OUTPUT_IMAGE_DIR)
print("Mask:", OUTPUT_MASK_DIR)