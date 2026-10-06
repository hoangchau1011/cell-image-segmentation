import os
import random

# ============================================================
# ĐƯỜNG DẪN
# ============================================================

IMAGE_DIR = r"D:\BraTS-GLI\processed\images"
MASK_DIR = r"D:\BraTS-GLI\processed\masks"

TRAIN_FILE = r"D:\BraTS-GLI\processed\train.txt"
VAL_FILE = r"D:\BraTS-GLI\processed\val.txt"


# ============================================================
# CẤU HÌNH
# ============================================================

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ============================================================
# LẤY DANH SÁCH SUBJECT
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
]

subjects = set()

for filename in image_files:

    # Ví dụ:
    # BraTS-GLI-00000-000_slice_049.png

    subject = filename.split("_slice_")[0]

    subjects.add(subject)

subjects = sorted(subjects)


print("=" * 60)
print("BraTS Patient-Level Split")
print("=" * 60)

print(f"Tổng số subject: {len(subjects)}")


# ============================================================
# SHUFFLE SUBJECT
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(subjects)


# ============================================================
# CHIA TRAIN / VALIDATION
# ============================================================

train_count = int(
    len(subjects) * TRAIN_RATIO
)

train_subjects = subjects[:train_count]
val_subjects = subjects[train_count:]


print(f"Train subjects: {len(train_subjects)}")
print(f"Val subjects  : {len(val_subjects)}")


# ============================================================
# TẠO SET ĐỂ TÌM NHANH
# ============================================================

train_subject_set = set(train_subjects)
val_subject_set = set(val_subjects)


# ============================================================
# TẠO DANH SÁCH TRAIN / VAL
# ============================================================

train_images = []
val_images = []


for filename in image_files:

    subject = filename.split("_slice_")[0]

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    # Chỉ lấy nếu mask tồn tại
    if not os.path.exists(mask_path):
        continue

    if subject in train_subject_set:
        train_images.append(filename)

    elif subject in val_subject_set:
        val_images.append(filename)


# ============================================================
# SORT
# ============================================================

train_images.sort()
val_images.sort()


# ============================================================
# GHI FILE
# ============================================================

with open(TRAIN_FILE, "w", encoding="utf-8") as f:

    for filename in train_images:
        f.write(filename + "\n")


with open(VAL_FILE, "w", encoding="utf-8") as f:

    for filename in val_images:
        f.write(filename + "\n")


# ============================================================
# KẾT QUẢ
# ============================================================

print()
print("=" * 60)
print("SPLIT HOÀN TẤT")
print("=" * 60)

print(f"Train images: {len(train_images)}")
print(f"Val images  : {len(val_images)}")

print()
print(f"Train file: {TRAIN_FILE}")
print(f"Val file  : {VAL_FILE}")

print("=" * 60)