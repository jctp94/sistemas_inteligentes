
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report
from anfis.model import AnfisNet
from anfis.membership import BellMembFunc
from torch.utils.data import TensorDataset, DataLoader

# Cargar datos
df = pd.read_csv("smart_mobility_dataset.csv")
df = df[["Traffic_Speed_kmh", "Road_Occupancy_%", "Traffic_Condition"]].dropna()

# Codificar etiquetas
label_map = {"Low": 0, "Medium": 1, "High": 2}
df["label"] = df["Traffic_Condition"].map(label_map)

# Normalizar entradas
scaler = MinMaxScaler()
X = scaler.fit_transform(df[["Traffic_Speed_kmh", "Road_Occupancy_%"]])
y = df["label"].values

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Crear DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Crear modelo ANFIS
mf_defs = [(BellMembFunc(2.0, 4.0, 0.25), BellMembFunc(2.0, 4.0, 0.75)) for _ in range(2)]
model = AnfisNet(2, mf_defs, 3)  # 2 entradas, 2 mfs por entrada, 3 salidas (clases)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# Evaluación
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).argmax(dim=1)
    print("\nReporte de clasificación ANFIS:")
    print(classification_report(y_test_tensor.numpy(), predictions.numpy(), target_names=["Low", "Medium", "High"]))
