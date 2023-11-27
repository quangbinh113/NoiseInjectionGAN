import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AIGAN import AIGAN
import torch.hub

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

    # Transformations for the COCO dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),  # Resize images to fit YOLO input
    ])

    # Load COCO dataset
    coco_dataset = datasets.CocoDetection(
        root='./data/coco/train2017',
        annFile='./data/coco/annotations/instances_train2017.json',
        transform=transform
    )
    dataloader = DataLoader(
        coco_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=True
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
        dataset_name="imagenet",
        is_targeted=False)

    aiGAN.train(dataloader, epochs)
