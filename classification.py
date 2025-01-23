import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class DiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        healthy_dir = os.path.join(root_dir, 'healthy')
        disease_dir = os.path.join(root_dir, 'disease')
        
        self.healthy_images = [os.path.join(healthy_dir, img) 
                                for img in os.listdir(healthy_dir) 
                                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.disease_images = [os.path.join(disease_dir, img) 
                                for img in os.listdir(disease_dir) 
                                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        self.images = self.healthy_images + self.disease_images
        self.labels = [0] * len(self.healthy_images) + [1] * len(self.disease_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_resnet101_model(num_classes=2):
    model = models.resnet101(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix using seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_model(model, dataloader):
    """
    Evaluate model and generate confusion matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print(classification_report(all_labels, all_preds, 
                                target_names=['Healthy', 'Disease']))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, 
                          classes=['Healthy', 'Disease'])

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.set_grad_enabled(phase == 'train'):
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100 * correct_predictions / total_predictions
            
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%')

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create datasets
    image_datasets = {
        'train': DiseaseDataset(root_dir='train', transform=data_transforms['train']),
        'val': DiseaseDataset(root_dir='test', transform=data_transforms['val'])
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2)
    }

    # Initialize model, loss, and optimizer
    model = get_resnet101_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer)

    # Evaluate and plot confusion matrix
    evaluate_model(trained_model, dataloaders['val'])

    # Save the model
    torch.save(trained_model.state_dict(), 'resnet101_disease_classifier.pth')

if __name__ == "__main__":
    main()