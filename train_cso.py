from ViT import VisionTransformer, ConvStemConfig
from cso import CSO
from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import yaml
import os
import glob
from IPython import embed

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './config.yml',
                    'Absolute path to config.py')

def main(argv):
    # Load the YAML file
    with open(FLAGS.config, 'r') as f:
        config = yaml.safe_load(f)

    top_k = 5
    os.makedirs(config["ckpt_path"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess dataset
    if config["dataset"] == 'imagenet1k':
        transform = transforms.Compose([
            transforms.Resize(256),                    # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),                # Crop the center 224x224 pixels
            transforms.RandomHorizontalFlip(),         # Randomly flip the image horizontally
            transforms.ToTensor(),                      # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            ])
    if config["dataset"] == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    if config["dataset"] == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)
    elif config["dataset"] == 'imagenet1k':
        train_dataset = torchvision.datasets.ImageNet(root='./data_imagenet1k', split='train', transform=transform)
        test_dataset = torchvision.datasets.ImageNet(root='./data_imagenet1k', split='val', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize the model
    conv_stem_configs = [
        ConvStemConfig(out_channels=48, kernel_size=3, stride=2),
        ConvStemConfig(out_channels=96, kernel_size=3, stride=2),
        ConvStemConfig(out_channels=192, kernel_size=3, stride=2),
        ConvStemConfig(out_channels=384, kernel_size=3, stride=2),
        ConvStemConfig(out_channels=384, kernel_size=1, stride=1),
    ]

    cso = CSO()
    cso.optimize(config, conv_stem_configs, 4, train_loader, -1, 1, 1000)
        
if __name__ == '__main__':
    app.run(main)
