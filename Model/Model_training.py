from dataloader import Dataloading
from CNN_model import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm





def train_model(model, train_data_loader, test_data_loader, loss_Function, optimizer, n_epoch, patience = 5):

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loss = []
    validation_loss = []
    train_acc = []
    validation_acc = []
    best_validation_loss = float('inf')

    for epoch in tqdm(range(n_epoch)):
        total_train_loss = 0
        total_validation_loss = 0
        correct_train = 0
        correct_validation = 0
        total_train_samples = 0
        total_validation_samples = 0
        current_patience = 0

        for inputs, labels in train_data_loader:
            y_pred = model(inputs)
            loss = loss_Function(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()# Updating paramaters based on Gradient.
            total_train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(y_pred.data, 1)  # Assuming you use PyTorch
            correct_train += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        for inputs, labels in test_data_loader:
            y_pred = model(inputs)
            loss = loss_Function(y_pred, labels)
            total_validation_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(y_pred.data, 1)  # Assuming you use PyTorch
            correct_validation += (predicted == labels).sum().item()
            total_validation_samples += labels.size(0)

        train_loss.append(total_train_loss / len(train_data_loader))
        validation_loss.append(total_validation_loss / len(test_data_loader))

        
        train_acc.append(100 * correct_train / total_train_samples)
        validation_acc.append(100 * correct_validation / total_validation_samples)

        print(f"Epoch {epoch}: Train Loss: {train_loss[-1]:.3f}, Train Acc: {train_acc[-1]:.2f}%, Validation Loss: {validation_loss[-1]:.3f}, Validation Acc: {validation_acc[-1]:.2f}%")

        if validation_loss[-1] < best_validation_loss:
            best_validation_loss = validation_loss[-1]
            current_patience = 0  # Reset patience counter if validation loss improves
            torch.save(model.state_dict(), "Digit_classifiaction_model.pth")  # Save best model
        else:
            current_patience += 1

        if current_patience >= patience:
            print(f"Early stopping triggered after {epoch} epochs with no improvement in validation loss")
            break  # Exit training loop if patience is exhausted

        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(validation_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Img.png")

        

if __name__ == "__main__":
  data = Dataloading('Nepali-Digit')
  train_data_loader, test_data_loader = data.Data_load()  # Get both loaders
  model = CNNModel()
  loss_Function = nn.CrossEntropyLoss()
  learning_rate = 0.001
  momentum = 0.9
  optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum)
  n_epoch = 20
  train_model(model, train_data_loader,test_data_loader, loss_Function, optimizer, n_epoch)
    

torch.save(model.state_dict(), "Digit_classifiaction_model.pth")
