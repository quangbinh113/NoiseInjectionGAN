import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AIGAN import AIGAN
from models import YourModelHere  # Import your desired model architecture

from torchvision.datasets import CocoDetection

if __name__ == "__main__":
    use_cuda = True
    image_nc = 3
    epochs = 600
    batch_size = 128
    C_TRESH = 0.5
    BOX_MIN = 0
    BOX_MAX = 1
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("CUDA Available: ", torch.cuda.is_available())
    print("Using device: ", device)

    MODEL_NAME = "your_model_for_mscoco"  # Change to a suitable model for MSCOCO
    model = YourModelHere()  # Load your model here
    model = model.to(device)
    model.eval()

    print("Successfully loaded target model ", MODEL_NAME)

    model_num_labels = 10  # Change this based on the MSCOCO dataset
    stop_epoch = 10

    # MSCOCO train dataset and dataloader declaration
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor()
    ])

    # Load MSCOCO dataset
    data_dir = './data/coco'  # Update this path
    ann_file = './data/coco/annotations/instances_train2017.json'  # Update this path
    coco_dataset = CocoDetection(root=data_dir, annFile=ann_file, transform=transform)
    dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    print("Training image examples: ", len(coco_dataset))

    aiGAN = AIGAN(device,
                  model,
                  model_num_labels,
                  image_nc,
                  stop_epoch,
                  BOX_MIN,
                  BOX_MAX,
                  C_TRESH,
                  dataset_name="coco",
                  is_targeted=False)

    aiGAN.train(dataloader, epochs)
