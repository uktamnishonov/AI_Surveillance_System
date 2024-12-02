import os
import torch
from torch.utils.data import DataLoader
from datasets.dataset_manager import ExpressionDataset
from models.expression_model import get_model
from training.trainer import train_model
from training.evaluator import test_model, evaluate_model
from utils.transforms import get_transforms

# Dataset paths
dataset_path = "/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1"
train_root = os.path.join(dataset_path, 'train')
test_root = os.path.join(dataset_path, 'test')

# Classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Data loaders
transform = get_transforms()
train_dataset = ExpressionDataset(train_root, classes, transform=transform)
test_dataset = ExpressionDataset(test_root, classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(pretrained=True, num_classes=len(classes)).to(device)

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

# Training
train_model(model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluation
test_model(model, test_loader, device)
evaluate_model(model, test_loader, device, classes)

# Save the model
torch.save(model.state_dict(), 'expression_model.pth')
print("Model saved as expression_model.pth")
