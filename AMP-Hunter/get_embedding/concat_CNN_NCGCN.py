import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

NCGCN_embedding_train = np.load('../AMP_dataset/embedding/NCGCN_embedding_train.npy')
NCGCN_embedding_val = np.load('../AMP_dataset/embedding/NCGCN_embedding_val.npy')
NCGCN_embedding_test = np.load('../AMP_dataset/embedding/NCGCN_embedding_test.npy')


CNN_embedding_train = np.load('../AMP_dataset/embedding/CNN_embedding_train.npy')
CNN_embedding_val = np.load('../AMP_dataset/embedding/CNN_embedding_val.npy')
CNN_embedding_test = np.load('../AMP_dataset/embedding/CNN_embedding_test.npy')


def load_labels_from_txt(path):
    labels = []
    values = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            first_col = parts[0]
            value = float(parts[-1])  

            if first_col.startswith("AMP"):
                label = 1
            elif first_col.startswith("NAMP"):
                label = 0
            else:
                continue  

            labels.append(label)  #classification task
            #values.append(value)  #regression task

    return labels



label_train = load_labels_from_txt('../AMP_dataset/raw/train.txt')
label_val   = load_labels_from_txt('../AMP_dataset/raw/val.txt')
label_test  = load_labels_from_txt('../AMP_dataset/raw/test.txt')

y_train = torch.tensor(label_train, dtype=torch.long)  #for regression task: float32
y_val   = torch.tensor(label_val, dtype=torch.long)
y_test  = torch.tensor(label_test, dtype=torch.long)

# concat NCGCN and CNN feature
embeddings_train = np.hstack((NCGCN_embedding_train, CNN_embedding_train))  
embeddings_val = np.hstack((NCGCN_embedding_val, CNN_embedding_val))  
embeddings_test = np.hstack((NCGCN_embedding_test, CNN_embedding_test))  

x_train = torch.tensor(embeddings_train, dtype=torch.float)
x_val = torch.tensor(embeddings_val, dtype=torch.float)
x_test = torch.tensor(embeddings_test, dtype=torch.float)

train_set = TensorDataset(x_train, y_train)
val_set = TensorDataset(x_val, y_val)
test_set = TensorDataset(x_test, y_test)


batch_size = 656
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
output_dim = 2


class Classifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(embeddings_train.shape[1], output_dim).to(device)  
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_class(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_train_labels = [] 
        all_train_preds = []

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred_labels = outputs.argmax(dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
            

            all_train_preds.extend(pred_labels.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_acc = correct / total
        val_acc,val_precision,val_sensitivity,val_spec,val_f1,val_labels,val_preds = evaluate_class(model, val_loader)


        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/100], Loss: {total_loss/len(train_loader):.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_spec:.4f}, Val F1: {val_f1:.4f}")


            

def evaluate_class(model, data_loader):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            TP = ((outputs.argmax(dim=1) == 1) & (labels == 1)).sum().item()
            TN = ((outputs.argmax(dim=1) == 0) & (labels == 0)).sum().item()
            FP = ((outputs.argmax(dim=1) == 1) & (labels == 0)).sum().item()
            FN = ((outputs.argmax(dim=1) == 0) & (labels == 1)).sum().item()

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0       
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0           
            precision   = TP / (TP + FP) if (TP + FP) > 0 else 0         
            recall      = sensitivity                                   
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            acc = correct / total


    return acc,precision,sensitivity,specificity,f1,all_labels,all_preds


train_class(model, train_loader, val_loader, criterion, optimizer, epochs=1000)
test_acc,test_precision,test_sensitivity,test_spec,test_f1,test_labels,test_preds = evaluate_class(model, test_loader)
print(f"Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_spec:.4f}, Test F1: {test_f1:.4f}")



def train_reg(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), labels)  
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        
        train_mse = total_loss / len(train_loader)

        
        val_mse,val_rmse,val_r2,val_mae = evaluate_mse(model, val_loader)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}, Val MAE: {val_mae:.4f}")


def evaluate_mse(model, data_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)

            loss = nn.MSELoss()(outputs.squeeze(), labels)  
            total_loss += loss.item()
            r2 = r2_score(outputs.squeeze().cpu(), labels.cpu())
            mae = mean_absolute_error(outputs.squeeze().cpu(), labels.cpu())


    mse = total_loss / len(data_loader)
    rmse = torch.sqrt(torch.tensor(mse))  
    return mse, rmse, r2, mae
'''
train_reg(model, train_loader, val_loader, criterion, optimizer, epochs = 1000)
test_mse, test_rmse, test_r2, test_mae = evaluate_mse(model, test_loader)
print(f"Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}")
'''