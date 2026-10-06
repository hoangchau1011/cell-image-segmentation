import kagglehub
import shutil
import os

print("=" * 50)
print("Đang tải Data Science Bowl 2018...")
print("=" * 50)

# Tải dataset từ Kaggle
dataset_path = kagglehub.dataset_download(
    "kmader/data-science-bowl-2018"
)

print("\nĐã tải xong!")
print("Dataset gốc nằm tại:")
print(dataset_path)

# Copy vào thư mục data trong project
destination = "./data/archive"

if not os.path.exists(destination):
    shutil.copytree(dataset_path, destination)
    print(f"\nĐã copy dataset sang {destination}")
else:
    print("\nThư mục data/archive đã tồn tại.")

print("\nHoàn tất!")