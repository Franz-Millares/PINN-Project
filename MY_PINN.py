import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def prepdata():
    file_path = r'C:\Users\ADMIN\Downloads\PINN code\Train-PINN-master\TrainPINNV3\train_Data\Fma-data.csv'
    data = pd.read_csv(file_path, index_col=0)

    time_units = data.index 
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    data = pd.DataFrame(data, columns=['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Weight', 'Force'])
    
    X = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']].values
    force = data['Force'].values

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    force_train, force_test = force[:train_size], force[train_size:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    force_train_tensor = torch.tensor(force_train, dtype=torch.float32).unsqueeze(1)
    force_test_tensor = torch.tensor(force_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_test_tensor, force_train_tensor, force_test_tensor, time_units, train_size

X_train_tensor, X_test_tensor, force_train_tensor, force_test_tensor, time_units, train_size = prepdata()

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x, return_features=False):
        features = x
        for layer in self.layers[:-1]:
            features = layer(features)
        if return_features:
            return features
        output = self.layers[-1](features)
        return output

layers = [6, 20, 20, 6]  # Output size matches the input feature size
model = PINN(layers)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

def pinn_loss(predictions, targets, force):
    data_loss = nn.MSELoss()(predictions, targets)
    
    force_pred = predictions[:, 0:1] * 3  # Directly multiply predictions by 3 to get force_pred
    physics_loss = nn.MSELoss()(force_pred, force)

    return data_loss + 0.1 * physics_loss

def train_model():
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        
        output = model(X_train_tensor)
        loss = pinn_loss(output, X_train_tensor, force_train_tensor)
        loss.backward()
        
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def eval_model():
    model.eval()
    with torch.no_grad():
        recon_train = model(X_train_tensor)
        recon_test = model(X_test_tensor)

    test_loss = pinn_loss(recon_test, X_test_tensor, force_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

    # Anomaly detection on training and test sets
    recon_train_np = recon_train.numpy()
    recon_test_np = recon_test.numpy()

    # Fit LOF on the training set
    lof_model = LocalOutlierFactor(n_neighbors=50)
    lof_model.fit(recon_train_np)

    # Detect anomalies
    anomalies_train = np.where(lof_model.fit_predict(recon_train_np) == -1)[0]
    anomalies_test = np.where(lof_model.fit_predict(recon_test_np) == -1)[0]

    # Calculate the original indices of anomalies in the test set
    original_anomaly_indices_test = anomalies_test + train_size

    # Find the corresponding time units for the detected anomalies
    anomalous_time_units_test = time_units[original_anomaly_indices_test]

    print(f"Anomalies detected in training set at indices: {anomalies_train}")
    print(f"Anomalies detected in test set at time units: {anomalous_time_units_test.values}")

    return X_train_tensor.numpy(), X_test_tensor.numpy(), anomalies_train, anomalies_test, anomalous_time_units_test

def plot_data_with_anomalies(X, anomalies, time_units, title):
    plt.figure(figsize=(12, 6))

    # Plot all data points
    plt.plot(time_units, X[:, 0], label='Normal Data', color='blue')

    # Highlight anomalies
    plt.scatter(time_units[anomalies], X[anomalies, 0], label='Anomalies', color='red', s=50, marker='x')

    plt.title(title)
    plt.xlabel('Time Units')
    plt.ylabel('Features Value')
    plt.legend()
    plt.show()

train_model()

# Train and evaluate the model
X_train_np, X_test_np, anomalies_train, anomalies_test, anomalous_time_units_test = eval_model()

# Plot training data with anomalies
plot_data_with_anomalies(X_train_np, anomalies_train, time_units[:train_size], 'Training Data with Anomalies')

# Plot test data with anomalies
plot_data_with_anomalies(X_test_np, anomalies_test, time_units[train_size:], 'Test Data with Anomalies')
