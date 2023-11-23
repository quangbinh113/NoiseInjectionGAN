import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage import io
from scipy.stats import entropy
import numpy as np
from torchvision.transforms.functional import to_tensor


# Hàm tính Inception Score (IS)
def calculate_inception_score(images, batch_size=32, splits=10):
    # Load pre-trained Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    inception_model.eval()
    
    def kl_divergence(p, q):
        return entropy(p, q)
    
    # Load and preprocess images
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    images = [transform(image) for image in images]
    
    dataloader = DataLoader(images, batch_size=batch_size)
    
    # Calculate IS
    scores = []
    for batch in dataloader:
        with torch.no_grad():
            batch_logits = inception_model(batch)
            probs = F.softmax(batch_logits, dim=1).cpu().numpy()
        
        p_y = np.mean(probs, axis=0)
        scores.append(np.exp(np.sum([p * np.log(p / q) for p, q in zip(p_y, p_y.mean())])))
    
    is_score = np.mean(scores)
    
    return is_score

# Hàm tính Frechet Inception Distance (FID)
def calculate_fid(images_real, images_fake, batch_size=32):
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    inception_model.eval()
    
    def extract_features(images):
        features = []
        dataloader = DataLoader(images, batch_size=batch_size)
        for batch in dataloader:
            with torch.no_grad():
                batch_features = inception_model(batch).detach().cpu().numpy()
                features.append(batch_features)
        return np.concatenate(features, axis=0)
    
    features_real = extract_features(images_real)
    features_fake = extract_features(images_fake)
    
    mu_real, sigma_real = np.mean(features_real, axis=0), np.cov(features_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(features_fake, axis=0), np.cov(features_fake, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean, _ = np.linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = np.dot(diff, diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid_score

# Đọc hai hình ảnh
image_real = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_1/id_1-0origin -label_8-classname_ships.jpg")
image_fake = io.imread("./preview/cifar10_mobilenetv2_x1_4/0_1/id_1-3adv-label_3_0.38840973377227783-classname_cats.jpg")

# Tính SSIM
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

# Tính chỉ số SSIM sử dụng hàm tích hợp trong PyTorch
ssim_score = 1 - F.mse_loss(image1_tensor, image2_tensor)

# Tính PSNR
mse = F.mse_loss(image1_tensor, image2_tensor)
psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

# Tính Inception Score (IS)

is_score = calculate_inception_score(image1 + image2)

# Tính Frechet Inception Distance (FID)
fid_score = calculate_fid(image1, image2)

print(f"SSIM: {ssim_score}")
print(f"PSNR: {psnr.item()}")
print(f"Inception Score (IS): {is_score}")
print(f"Frechet Inception Distance (FID): {fid_score}")
