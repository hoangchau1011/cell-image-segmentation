import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        list_file,
        image_size=(256, 256),
        boundary_width=1
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.boundary_width = boundary_width

        # ==================================================
        # Đọc danh sách ảnh
        # ==================================================

        with open(list_file, "r", encoding="utf-8") as f:

            self.image_files = [
                line.strip()
                for line in f
                if line.strip()
            ]

        print(
            f"Loaded {len(self.image_files)} images from:"
        )

        print(list_file)

    # ======================================================
    # Số lượng mẫu
    # ======================================================

    def __len__(self):

        return len(self.image_files)

    # ======================================================
    # Tạo mask 3 class
    #
    # 0 = Background
    # 1 = Tumor
    # 2 = Boundary
    # ======================================================

    def create_three_class_mask(self, binary_mask):

        # ----------------------------------------------
        # Đảm bảo mask là binary
        # ----------------------------------------------

        tumor = (binary_mask > 0).astype(np.uint8)

        # ----------------------------------------------
        # Tạo kernel
        # ----------------------------------------------

        kernel_size = self.boundary_width * 2 + 1

        kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8
        )

        # ----------------------------------------------
        # Erode tumor
        # ----------------------------------------------

        eroded = cv2.erode(
            tumor,
            kernel,
            iterations=1
        )

        # ----------------------------------------------
        # Boundary = tumor - vùng tumor bên trong
        # ----------------------------------------------

        boundary = tumor - eroded

        # ----------------------------------------------
        # Khởi tạo mask 3 class
        #
        # 0 = background
        # 1 = tumor
        # 2 = boundary
        # ----------------------------------------------

        mask_3class = np.zeros(
            tumor.shape,
            dtype=np.uint8
        )

        # Tumor
        mask_3class[tumor > 0] = 1

        # Boundary
        mask_3class[boundary > 0] = 2

        return mask_3class

    # ======================================================
    # Lấy một sample
    # ======================================================

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        # ==================================================
        # Đường dẫn image
        # ==================================================

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        # ==================================================
        # Đường dẫn mask
        # ==================================================

        mask_path = os.path.join(
            self.mask_dir,
            filename
        )

        # ==================================================
        # Đọc ảnh MRI
        # ==================================================

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ==================================================
        # Đọc mask
        # ==================================================

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ==================================================
        # Kiểm tra
        # ==================================================

        if image is None:

            raise FileNotFoundError(
                f"Không đọc được image: {image_path}"
            )

        if mask is None:

            raise FileNotFoundError(
                f"Không đọc được mask: {mask_path}"
            )

        # ==================================================
        # Resize image
        # ==================================================

        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        # ==================================================
        # Resize mask
        # ==================================================

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        # ==================================================
        # Normalize image
        # ==================================================

        image = image.astype(
            np.float32
        ) / 255.0

        # ==================================================
        # Tạo mask 3 class
        # ==================================================

        mask = self.create_three_class_mask(
            mask
        )

        # ==================================================
        # Thêm channel cho image
        #
        # (H, W)
        #      ↓
        # (1, H, W)
        # ==================================================

        image = np.expand_dims(
            image,
            axis=0
        )

        # ==================================================
        # NumPy → PyTorch
        # ==================================================

        image = torch.from_numpy(
            image
        ).float()

        # ==================================================
        # Mask phải là LONG
        #
        # CrossEntropyLoss yêu cầu:
        #
        # [H, W]
        #
        # và giá trị:
        # 0 / 1 / 2
        # ==================================================

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask