import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from unet_model import UNet

# ==================================================
# CẤU HÌNH
# ==================================================

MODEL_WEIGHTS = "unet_pet_epoch_5.pth"

TEST_IMAGE_PATH = "./data/oxford-iiit-pet/images/Abyssinian_1.jpg"


def main():

    # ==================================================
    # KHỞI TẠO MÔ HÌNH
    # ==================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = UNet(
        in_channels=3,
        out_classes=3
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_WEIGHTS,
            map_location=device
        )
    )

    model.eval()

    print("Đã tải trọng số thành công!")

    # ==================================================
    # ĐỌC ẢNH
    # ==================================================

    img = cv2.imread(TEST_IMAGE_PATH)

    if img is None:
        print("Không đọc được ảnh!")
        return

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # ==================================================
    # TIỀN XỬ LÝ
    # ==================================================

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(img_rgb)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # ==================================================
    # DỰ ĐOÁN
    # ==================================================

    with torch.no_grad():

        output = model(input_tensor)

        predicted_mask = torch.argmax(
            output,
            dim=1
        )

        predicted_mask = (
            predicted_mask
            .squeeze(0)
            .cpu()
            .numpy()
        )

    print("Các nhãn dự đoán:",
          np.unique(predicted_mask))

    # ==================================================
    # TẠO OVERLAY
    # ==================================================

    img_resized = cv2.resize(
        img_rgb,
        (256, 256)
    )

    color_mask = np.zeros_like(img_resized)

    # Theo dataset.py:
    # 0 = Pet
    # 1 = Border
    # 2 = Background

    # Pet -> Đỏ
    color_mask[predicted_mask == 0] = [255, 0, 0]

    # Border -> Vàng
    color_mask[predicted_mask == 1] = [255, 255, 0]

    # Background -> Đen
    color_mask[predicted_mask == 2] = [0, 0, 0]

    overlay = cv2.addWeighted(
        img_resized,
        0.7,
        color_mask,
        0.5,
        0
    )

    # ==================================================
    # HIỂN THỊ KẾT QUẢ
    # ==================================================

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_resized)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="viridis")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()