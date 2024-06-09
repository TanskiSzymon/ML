# Wymagane jest pobranie danych z:
# https://www.kaggle.com/datasets/luluw8071/brain-tumor-mri-datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Sprawdzenie dostępności MPS (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Definicja architektury modelu
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Inicjalizacja modelu i przeniesienie na odpowiednie urządzenie
model = SimpleCNN().to(device)

# Definicja funkcji straty
criterion = nn.NLLLoss()

# Optymalizator z adaptacyjnym współczynnikiem uczenia i momentum
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funkcja do wczytywania obrazów z folderu
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['yes', 'no']:
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            try:
                with Image.open(img_path).convert('RGB') as img:
                    img = img.resize((64, 64))
                    img_array = np.asarray(img) / 255.0
                    images.append(img_array)
                    labels.append(1 if label == 'yes' else 0)
            except IOError:
                print(f"Error opening image {img_path}")
    images = np.array(images)
    labels = np.array(labels)
    # Konwersja na tensory PyTorch
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

# Wczytywanie danych
train_images, train_labels = load_images_from_folder('Brain neural/train')
test_images, test_labels = load_images_from_folder('Brain neural/test')

# Tworzenie DataLoader
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Funkcja treningowa z wczesnym zatrzymaniem
def train_model(train_loader, model, criterion, optimizer, epochs=10, early_stop_thresh=0.01):
    model.train()
    best_loss = float('inf')
    train_losses = []
    accuracies = []
    weights_history = {name: [] for name, _ in model.named_parameters() if 'weight' in name}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(train_loader)} batches in Epoch {epoch+1}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

        # Zapisywanie wartości wag w poszczegolnych epokach
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights_history[name].append(param.detach().cpu().numpy().copy())

        if epoch_loss < early_stop_thresh:
            print(f"Stopping early at epoch {epoch+1}")
            break

    # Wykres błędu na zbiorze uczącym
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    # Wykres accuracy na zbiorze uczącym
    plt.figure()
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.show()

    return weights_history

# Funkcja do rysowania wag poszczególnych warstw
def plot_weights(weights_history):
    for name, weights in weights_history.items():
        plt.figure()
        num_neurons = weights[0].size  # liczba neuronów w danej warstwie
        for i in range(min(10, num_neurons)):  # rysowanie maksymalnie 10 wag dla czytelności
            weight_values = [weights[epoch].flat[i] for epoch in range(len(weights))]
            plt.plot(range(1, len(weights) + 1), weight_values, label=f'Weight {i}')
        plt.xlabel('Epochs')
        plt.ylabel('Weight Value')
        plt.title(f'Weight {name} Change over Epochs')
        plt.legend(loc='upper right')
        plt.show()

# Trenowanie modelu
weights_history = train_model(train_loader, model, criterion, optimizer, epochs=10)

# Ewaluacja modelu
def evaluate_model(test_images, test_labels, model):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        images, labels = test_images.to(device), test_labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    print(f'Accuracy: {100 * correct / total}%')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

# Ewaluacja modelu na danych testowych
evaluate_model(test_images, test_labels, model)

# Rysowanie wag modelu
plot_weights(weights_history)
