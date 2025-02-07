import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(35),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transforms)
    # Although not used during training
    test_dataset  = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    # Return only train and valid loaders along with the class-to-index mapping
    return train_loader, valid_loader, train_dataset.class_to_idx

def create_model(arch, hidden_units, learning_rate, device, num_classes):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        # Default to vgg16 if architecture not recognized
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    steps = 0
    print_every = 5
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy = 0
                with torch.no_grad():
                    for inputs_val, labels_val in valid_loader:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        logps_val = model.forward(inputs_val)
                        batch_loss = criterion(logps_val, labels_val)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_val.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}")
                train_loss = 0
                model.train()

def save_checkpoint(model, save_dir, arch, class_to_idx, hidden_units, learning_rate, epochs, optimizer):
    checkpoint = {
        'architecture': arch,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }
    save_path = os.path.join(save_dir, "checkpoint.pth")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint.get('architecture', 'vgg16')
    hidden_units = checkpoint.get('hidden_units', 512)
    learning_rate = checkpoint.get('learning_rate', 0.0025)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier from checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = class_to_idx

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if checkpoint['optimizer_state'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    epochs = checkpoint.get('epochs', 0)
    
    return model, optimizer, epochs

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    image = image.resize((new_width, new_height))
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image, dtype=torch.float32)

def predict(image_path, model, topk=5):
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probs, top_indices = probabilities.topk(topk)
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    return top_probs, top_classes
