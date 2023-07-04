from ViR import VisionRWKV, ConvStemConfig
from ViT import VisionTransformer
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

    if config["backbone"].lower() == 'vir':
        model = VisionRWKV(
            config, 
            config["image_size"], 
            config["patch_size"], 
            config["n_embd"], 
            config["num_classes"],
            conv_stem_configs).to(device)
    else:
        model = VisionTransformer(
            config, 
            config["image_size"], 
            config["patch_size"], 
            config["n_embd"], 
            config["num_classes"],
            conv_stem_configs).to(device)
    model = torch.jit.script(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    parameters_with_decay = []
    parameters_without_decay = []

    for name, parameter in model.named_parameters():
        if 'weight' in name:
            parameters_with_decay.append(parameter)
        else:
            parameters_without_decay.append(parameter)

    optimizer = optim.AdamW(
        [{'params': parameters_with_decay, 'weight_decay': float(config["weight_decay"])},
        {'params': parameters_without_decay, 'weight_decay': 0.0}], 
        lr=float(config["learning_rate"]), 
        betas=(float(config["beta_1"]), float(config["beta_2"])), 
        eps=float(config["adam_eps"]), 
        amsgrad=False,
        fused=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Training loop
    total_steps = len(train_loader)

    # Check if a checkpoint exists for resuming training
    resume = config["resume"]
    checkpoints = glob.glob(os.path.join(config["ckpt_path"], "model_epoch_*.pt"))
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by epoch number

    if resume and len(checkpoints) > 0:
        checkpoints_to_resume = checkpoints[-1]
        print(F"Resuming from {checkpoints_to_resume}")
        checkpoint = torch.load(os.path.join(config["ckpt_path"], checkpoints_to_resume))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        print("Training from scratch.")
        epoch = 0

    for epoch in range(config["num_epochs"]):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, include_head=True)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])
            optimizer.step()

            if (i+1) % config["print_step"] == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")
            scheduler.step()
        # Save checkpoint after every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        # Save the model after each epoch
        checkpoint_path = os.path.join(config["ckpt_path"], f"model_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Remove older checkpoints, keeping only top_k checkpoints
        checkpoints = glob.glob(os.path.join(config["ckpt_path"], "model_epoch_*.pt"))
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by epoch number
        checkpoints_to_delete = checkpoints[:-top_k]
        for checkpoint in checkpoints_to_delete:
            os.remove(checkpoint)

        if epoch % config["eval_epoch"] == 0:
            # Test the model
            print("Evaluating the model.")
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images, include_head=True)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(f"Test Accuracy: {accuracy:.2f}%")
        
if __name__ == '__main__':
    app.run(main)
