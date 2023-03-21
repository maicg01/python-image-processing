import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

# Tải mô hình EDSR x4
model = torchvision.models.video.edsr(scale_factor=4)
model.eval()

# Chuyển mô hình sang thiết bị GPU nếu có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Đọc ảnh khuôn mặt bị mờ
image_path = "/home/maicg/Documents/Me/oop_faceRecog/data_test/16_03_2023/unknow/17_03_2023-11:59:08_0.38.jpg"
image = Image.open(image_path)

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

input_image = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    output_image = model(input_image)

# Chuyển ảnh đầu ra thành định dạng PIL Image
output_image = output_image.squeeze().cpu().detach()
output_image = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])(output_image)
output_image = transforms.ToPILImage()(output_image)

output_image_path = "face_super_resolved.jpg"
output_image.save(output_image_path)