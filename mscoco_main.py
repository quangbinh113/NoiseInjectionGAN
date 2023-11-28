import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AIGAN import AIGAN
import torch.hub
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor


class CustomCocoDataset(Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __getitem__(self, idx):
        image, annotation = self.coco_dataset[idx]
        # Extract label from annotation
        # For simplicity, let's say we take the category ID of the first object in each image
        label = annotation[0]['category_id'] if len(annotation) > 0 else 0
        return image, label

    def __len__(self):
        return len(self.coco_dataset)



if __name__ == "__main__":
    use_cuda=True
    image_nc=3
    epochs = 600
    batch_size = 128 
    C_TRESH = 0.5
    BOX_MIN = 0
    BOX_MAX = 1

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("CUDA Available: ",torch.cuda.is_available())
    print("Using device: ", device)

    # Load YOLO model (use an appropriate version of YOLO)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model = yolo_model.to(device)
    yolo_model.eval()

    print("Successfully loaded YOLO model")

    model_num_labels = 80  # Number of classes in COCO dataset
    stop_epoch = 10

    # Define transformations for MSCOCO
    coco_transforms = Compose([
        Resize((224, 224)),  # Resize the image to 224x224 (or any size your model expects)
        ToTensor(),  # Convert the image to a PyTorch tensor
        # Add any other transformations your model might require
    ])

    coco_dataset = CustomCocoDataset(
        root='/kaggle/input/coco-2017-dataset/coco2017/train2017',
        annotation='/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json',
        transform=coco_transforms
    )

    coco_dataset = CustomCocoDataset(coco_dataset)

    # Transformations for the COCO dataset
    # transform = transforms.Compose([
    #     # transforms.RandomResizedCrop(size=299),#, scale=(1., 1.0)
    #     transforms.RandomResizedCrop(size=224),#, scale=(1., 1.0)
    #     transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    #     # transforms.RandomRotation(degrees=15),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #WARN don`t do it!!
    # ])

    # # Load COCO dataset
    # coco_dataset = CustomCocoDataset(
    #     root='/kaggle/input/coco-2017-dataset/coco2017/train2017',
    #     annotation='/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json',
    #     transform=transform
    # )

    dataloader = DataLoader(
        coco_dataset, 
        batch_size=batch_size, 
        shuffle=True,  
        drop_last=True,
        collate_fn=lambda x: x 
    )

    print("Training image examples: ", len(coco_dataset))

    aiGAN = AIGAN(
        device,
        yolo_model,
        model_num_labels,
        image_nc,
        stop_epoch,
        BOX_MIN,
        BOX_MAX,
        C_TRESH,
        dataset_name="coco",
        is_targeted=False)

    aiGAN.train(dataloader, epochs)