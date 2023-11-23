import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from skimage import io

# Đọc hai hình ảnh
image1 = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_1/id_1-0origin -label_8-classname_ships.jpg")
image2 = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_1/id_1-3adv-label_3_0.38840973377227783-classname_cats.jpg")
#image1 = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_10/id_1-0origin -label_8-classname_ships.jpg")
#image2 = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_10/id_1-3adv-label_3_0.5480353236198425-classname_cats.jpg")

# Chuyển hình ảnh thành tensor PyTorch
image1_tensor = to_tensor(image1).unsqueeze(0)
image2_tensor = to_tensor(image2).unsqueeze(0)

# Chuyển sang kiểu dữ liệu float32 và chuẩn hóa giá trị về khoảng [0, 1]
image1_tensor = image1_tensor.type(torch.float32) / 255.0
image2_tensor = image2_tensor.type(torch.float32) / 255.0

# Tính PSNR
mse = F.mse_loss(image1_tensor, image2_tensor)
psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

print(f"PSNR: {psnr.item()}")
