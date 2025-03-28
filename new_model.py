import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define Data Directory (Change this if your data is stored elsewhere)
data_dir = r"C:\Users\prem thakkar\OneDrive\Desktop\sd-198\images"

# Define Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20  # Adjust as needed
num_classes = 198  # Your dataset has 198 disease classes

# Data Transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load Dataset
dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ["train", "val"]}
dataloaders = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val"]}

# Load Pretrained Model (EfficientNet)
model = models.efficientnet_b3(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify for 198 classes

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Loop
def train_model():
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(dataloaders['train'])}, Train Acc: {train_acc}%")

    # Save Model
    torch.save(model.state_dict(), "skin_disease_model.pth")
    print("Training Complete. Model Saved!")

# Start Training
train_model()
