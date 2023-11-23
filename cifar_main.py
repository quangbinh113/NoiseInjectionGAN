import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AIGAN import AIGAN
from models import CIFAR_target_net

if __name__ == "__main__":
    use_cuda=True
    image_nc=3
    epochs = 600
    batch_size = 128 
    #C_TRESH =  0.3
    C_TRESH =  0.5
    BOX_MIN = 0
    BOX_MAX = 1
    # increase to speed up training phase but be carefull of how model is gonna be converged
    # Define what device we are using
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("CUDA Available: ",torch.cuda.is_available())
    print("Using device: ", device)

    #Gene: input tam hinh goc  >> pert : encoder sau conv2d lam nho cai hinh down scaling >> up sampling  decode lam lon cai hinh lai >> hinh goc
    #Dicri: input  Output G (pert) >> d_fake_probs output xac xuat D , D_fake : 1 tam hinh nha ra 1 phantu ( kiem tra pert co thay doi dc label ko)

    # target_model_path = r"C:\Users\nguye\OneDrive\Máy tính\SBC\Deepfake_GAN\targermodel\cifar10_resnet56-187c023a.pt"
    # model = CIFAR_target_net().to(device)
    # model.load_state_dict(torch.load(target_model_path))
    # model = torch.load(target_model_path).to(device)

    #MODEL_NAME = "cifar10_resnet56"
    #MODEL_NAME = "cifar10_vgg19_bn"
    #MODEL_NAME = "cifar10_mobilenetv2_x1_4"
    #MODEL_NAME = "cifar10_shufflenetv2_x2_0"
    MODEL_NAME = "cifar10_repvgg_a2"
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", MODEL_NAME, pretrained=True)
    model = model.to(device)
    model.eval()

    print("Successfully loaded target model ", MODEL_NAME)

    model_num_labels = 10
    stop_epoch = 10


    # MNIST train dataset and dataloader declaration
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5) ]) # WARN don't do it
    transform = transforms.ToTensor()

    cifar_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    print("training image examples: ", cifar_dataset.__len__())
    # target = 4
    # labels = dataloader.dataset.targets
    # targets = torch.zeros_like(labels) + target 
    # dataloader.dataset.targets = targets

    aiGAN = AIGAN(device,
                    model,
                    model_num_labels,
                    image_nc,
                    stop_epoch,
                    BOX_MIN,
                    BOX_MAX,
                    C_TRESH,
                    dataset_name="cifar10",
                    # is_targeted=True)
                    is_targeted=False)

    aiGAN.train(dataloader, epochs)