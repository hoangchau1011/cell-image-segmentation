import os
import cv2
import numpy as np
import nibabel as nib

from tqdm import tqdm


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

DATASET_DIR = (
    r"D:\BraTS-GLI\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_IMAGE_DIR = r"D:\BraTS-GLI\processed\images"
OUTPUT_MASK_DIR = r"D:\BraTS-GLI\processed\masks"


# ============================================================
# 2. CẤU HÌNH
# ============================================================

# Kích thước ảnh đầu ra cho U-Net
IMAGE_SIZE = (256, 256)

# Chỉ lưu những slice có tumor
ONLY_TUMOR_SLICES = True


# ============================================================
# 3. TẠO THƯ MỤC OUTPUT
# ============================================================

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


# ============================================================
# 4. CHUẨN HÓA MRI
# ============================================================

def normalize_image(image):
    """
    Chuẩn hóa ảnh MRI về khoảng 0-255.

    Chỉ sử dụng các voxel khác 0 để tránh
    vùng nền ảnh hưởng đến normalization.
    """

    image = image.astype(np.float32)

    # Các voxel thuộc vùng não
    non_zero = image[image > 0]

    if len(non_zero) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Percentile giúp giảm ảnh hưởng của outlier
    p1 = np.percentile(non_zero, 1)
    p99 = np.percentile(non_zero, 99)

    image = np.clip(image, p1, p99)

    image = (image - p1) / (p99 - p1 + 1e-8)

    image = image * 255.0

    return image.astype(np.uint8)


# ============================================================
# 5. CHUYỂN MASK BRATS THÀNH BINARY MASK
# ============================================================

def create_binary_mask(mask):
    """
    BraTS:
        0 = background
        1 = tumor class
        2 = tumor class
        4 = tumor class

    Chuyển thành:
        0 = background
        1 = tumor
    """

    binary_mask = np.zeros_like(mask, dtype=np.uint8)

    binary_mask[mask > 0] = 1

    return binary_mask


# ============================================================
# 6. LẤY DANH SÁCH SUBJECT
# ============================================================

subjects = [
    name
    for name in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, name))
]

subjects.sort()

print("=" * 60)
print("BraTS preprocessing")
print("=" * 60)

print(f"Dataset: {DATASET_DIR}")
print(f"Số subject: {len(subjects)}")
print(f"Output images: {OUTPUT_IMAGE_DIR}")
print(f"Output masks : {OUTPUT_MASK_DIR}")
print("=" * 60)


# ============================================================
# 7. XỬ LÝ TỪNG SUBJECT
# ============================================================

total_saved = 0
total_skipped = 0


for subject in tqdm(subjects, desc="Processing subjects"):

    subject_dir = os.path.join(DATASET_DIR, subject)

    # --------------------------------------------------------
    # File T2-FLAIR
    # --------------------------------------------------------

    flair_path = os.path.join(
        subject_dir,
        f"{subject}-t2f.nii.gz"
    )

    # --------------------------------------------------------
    # File segmentation
    # --------------------------------------------------------

    seg_path = os.path.join(
        subject_dir,
        f"{subject}-seg.nii.gz"
    )

    # Kiểm tra file
    if not os.path.exists(flair_path):
        print(f"\nKhông tìm thấy FLAIR: {subject}")
        continue

    if not os.path.exists(seg_path):
        print(f"\nKhông tìm thấy segmentation: {subject}")
        continue

    # --------------------------------------------------------
    # Đọc NIfTI
    # --------------------------------------------------------

    flair_nii = nib.load(flair_path)
    seg_nii = nib.load(seg_path)

    flair = flair_nii.get_fdata()
    seg = seg_nii.get_fdata()

    # --------------------------------------------------------
    # Kiểm tra kích thước
    # --------------------------------------------------------

    if flair.shape != seg.shape:

        print(
            f"\nShape không khớp: {subject}"
        )

        print(
            f"FLAIR: {flair.shape}"
        )

        print(
            f"SEG: {seg.shape}"
        )

        continue

    # --------------------------------------------------------
    # Chuẩn hóa FLAIR
    # --------------------------------------------------------

    flair = normalize_image(flair)

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    seg = create_binary_mask(seg)

    # --------------------------------------------------------
    # Duyệt 155 slice
    # --------------------------------------------------------

    for slice_idx in range(flair.shape[2]):

        image_slice = flair[:, :, slice_idx]

        mask_slice = seg[:, :, slice_idx]

        # ----------------------------------------------------
        # Bỏ slice không có tumor
        # ----------------------------------------------------

        if ONLY_TUMOR_SLICES:

            if np.sum(mask_slice) == 0:

                total_skipped += 1

                continue

        # ----------------------------------------------------
        # Resize ảnh
        # ----------------------------------------------------

        image_slice = cv2.resize(
            image_slice,
            IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Resize mask
        # ----------------------------------------------------

        mask_slice = cv2.resize(
            mask_slice,
            IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Tên file
        # ----------------------------------------------------

        filename = (
            f"{subject}_slice_{slice_idx:03d}.png"
        )

        image_path = os.path.join(
            OUTPUT_IMAGE_DIR,
            filename
        )

        mask_path = os.path.join(
            OUTPUT_MASK_DIR,
            filename
        )

        # ----------------------------------------------------
        # Lưu
        # ----------------------------------------------------

        cv2.imwrite(
            image_path,
            image_slice
        )

        cv2.imwrite(
            mask_path,
            mask_slice
        )

        total_saved += 1


# ============================================================
# 8. KẾT QUẢ
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING HOÀN TẤT")
print("=" * 60)

print(f"Số subject       : {len(subjects)}")
print(f"Slice đã lưu     : {total_saved}")
print(f"Slice bị bỏ      : {total_skipped}")

print()
print("Images:")
print(OUTPUT_IMAGE_DIR)

print()
print("Masks:")
print(OUTPUT_MASK_DIR)

print("=" * 60)