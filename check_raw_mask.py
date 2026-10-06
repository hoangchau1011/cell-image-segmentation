import os
import glob
import nibabel as nib
import numpy as np


# ==========================================
# ĐƯỜNG DẪN DATASET GỐC
# ==========================================

DATA_DIR = r"D:\BraTS-GLI"


# ==========================================
# TÌM FILE SEGMENTATION
# ==========================================

seg_files = glob.glob(
    os.path.join(DATA_DIR, "**", "*-seg.nii.gz"),
    recursive=True
)


print("Số file mask tìm thấy:", len(seg_files))

if len(seg_files) == 0:
    print("\nKHÔNG tìm thấy file *-seg.nii.gz")
    print("Hãy kiểm tra lại đường dẫn DATA_DIR.")
    exit()


# ==========================================
# KIỂM TRA FILE ĐẦU TIÊN
# ==========================================

seg_path = seg_files[0]

print("\nFile đang kiểm tra:")
print(seg_path)


# ==========================================
# ĐỌC MASK
# ==========================================

nii = nib.load(seg_path)

mask = nii.get_fdata()

print("\nShape:")
print(mask.shape)

print("\nKiểu dữ liệu:")
print(mask.dtype)


# ==========================================
# LẤY CÁC LABEL
# ==========================================

unique_values = np.unique(mask)

print("\nCác giá trị label trong mask:")
print(unique_values)


# ==========================================
# ĐẾM SỐ VOXEL MỖI LABEL
# ==========================================

print("\nSố voxel của từng label:")

for value in unique_values:

    count = np.sum(mask == value)

    print(
        f"Label {value}: {count} voxel"
    )