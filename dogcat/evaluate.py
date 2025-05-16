import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

from model import get_model
from utils import get_test_loader

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate the model on the test set
    
    Args:
        model: PyTorch model
        test_loader: PyTorch data loader for testing
        criterion: Loss function
        device (str): Device to use for evaluation
        
    Returns:
        test_loss: Test loss
        test_acc: Test accuracy
        all_preds: All predictions
        all_labels: All labels
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    all_preds = []
    all_labels = []
    
    # Iterate over data
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            # Forward
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(all_preds, all_labels, classes):
    """
    Plot confusion matrix
    
    Args:
        all_preds: All predictions
        all_labels: All labels
        classes (list): List of class names
    """
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_predictions(model, test_loader, classes, num_samples=8, device='cuda'):
    """
    Visualize predictions
    
    Args:
        model: PyTorch model
        test_loader: PyTorch data loader for testing
        classes (list): List of class names
        num_samples (int): Number of samples to visualize
        device (str): Device to use for evaluation
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
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
        
        # Set title color based on prediction correctness
        color = 'green' if preds[i] == labels[i] else 'red'
        title = f'True: {classes[labels[i]]}\nPred: {classes[preds[i][0]]}\nProb: {probs[i][0]:.2f}'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data directory
    data_dir = '/Users/sonnguyen/Downloads/dog-cat-full-dataset-master/data'
    
    # Get test loader
    test_loader = get_test_loader(data_dir, batch_size=32)
    
    # Get model
    model = get_model()
    model.load_state_dict(torch.load('dogcat_model.pth', map_location=device))
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate model
    _, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device=device)
    
    # Plot confusion matrix
    classes = ['cat', 'dog']
    plot_confusion_matrix(all_preds, all_labels, classes)
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Visualize predictions
    print('\nVisualizing predictions:')
    visualize_predictions(model, test_loader, classes, device=device)

if __name__ == '__main__':
    main()
