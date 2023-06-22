from ViR import VisionRWKV
from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import deepspeed
from deepspeed.ops.adam import FusedAdam
import yaml
from IPython import embed

FLAGS = flags.FLAGS
flags.DEFINE_string('config', './config.yml',
                    'Absolute path to config.py')

def main(argv):
    # Load the YAML file
    with open(FLAGS.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize the model
    model = VisionRWKV(config, config["image_size"], config["patch_size"], config["n_embd"], config["num_classes"]).to(device)
    model = torch.jit.script(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = FusedAdam(model.parameters(), 
                          lr=config["learning_rate"], 
                          betas=(config["beta_1"], config["beta_2"]), 
                          eps=config["adam_eps"], 
                          bias_correction=True, 
                          adam_w_mode=False, 
                          weight_decay=0, 
                          amsgrad=False)

    # Training loop
    total_steps = len(train_loader)
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
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        
if __name__ == '__main__':
    app.run(main)
