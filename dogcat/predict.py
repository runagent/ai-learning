import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

from model import get_model

def predict_image(model, image_path, device='cuda'):
    """
    Predict the class of an image
    
    Args:
        model: PyTorch model
        image_path (str): Path to the image
        device (str): Device to use for inference
        
    Returns:
        pred_class (int): Predicted class (0 for cat, 1 for dog)
        prob (float): Probability of the prediction
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
    
    return pred_class, prob

def display_prediction(image_path, pred_class, prob):
    """
    Display the image with its prediction
    
    Args:
        image_path (str): Path to the image
        pred_class (int): Predicted class (0 for cat, 1 for dog)
        prob (float): Probability of the prediction
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    
    # Set title
    classes = ['cat', 'dog']
    title = f'Prediction: {classes[pred_class]} ({prob:.2f})'
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict dog or cat from an image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('--model_path', type=str, default='dogcat_model.pth', help='Path to the model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Make prediction
    pred_class, prob = predict_image(model, args.image_path, device=device)
    
    # Display prediction
    classes = ['cat', 'dog']
    print(f'Prediction: {classes[pred_class]}')
    print(f'Probability: {prob:.4f}')
    
    # Display the image with its prediction
    display_prediction(args.image_path, pred_class, prob)

if __name__ == '__main__':
    main()
