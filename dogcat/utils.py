import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class DogCatDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        """
        Args:
            img_paths (list): List of image paths
            labels (list): List of labels (0 for cat, 1 for dog)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, random_state=42):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size
        val_split (float): Validation split ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        train_loader, val_loader: PyTorch data loaders for training and validation
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image paths and labels
    train_dir = os.path.join(data_dir, 'train')
    
    cat_dir = os.path.join(train_dir, 'cats')
    dog_dir = os.path.join(train_dir, 'dogs')
    
    cat_paths = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    dog_paths = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    all_paths = cat_paths + dog_paths
    all_labels = [0] * len(cat_paths) + [1] * len(dog_paths)
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=val_split, random_state=random_state, stratify=all_labels
    )
    
    # Create datasets
    train_dataset = DogCatDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DogCatDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def get_test_loader(data_dir, batch_size=32):
    """
    Create test data loader
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size
        
    Returns:
        test_loader: PyTorch data loader for testing
    """
    # Define transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image paths and labels
    test_dir = os.path.join(data_dir, 'test')
    
    cat_dir = os.path.join(test_dir, 'cats')
    dog_dir = os.path.join(test_dir, 'dogs')
    
    cat_paths = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    dog_paths = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    all_paths = cat_paths + dog_paths
    all_labels = [0] * len(cat_paths) + [1] * len(dog_paths)
    
    # Create dataset
    test_dataset = DogCatDataset(all_paths, all_labels, transform=test_transform)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader

def visualize_samples(data_loader, classes, num_samples=8):
    """
    Visualize samples from the data loader
    
    Args:
        data_loader: PyTorch data loader
        classes (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    # Get a batch of images
    images, labels = next(iter(data_loader))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean
    
    # Plot images
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
