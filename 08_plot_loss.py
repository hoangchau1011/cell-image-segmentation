import matplotlib.pyplot as plt

# Dữ liệu Loss trích xuất từ quá trình huấn luyện 5 epochs
epochs = [1, 2, 3, 4, 5]
train_loss = [0.6194, 0.4457, 0.3944, 0.3616, 0.3309]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

plt.title('Biểu đồ Hàm mất mát (Training Loss) qua các chu kỳ', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)

# Lưu thành tệp ảnh để chèn vào báo cáo
plt.savefig('loss_curve.png', dpi=300)
print("Đã xuất biểu đồ thành công: loss_curve.png")