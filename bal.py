import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importar SMOTE
from imblearn.over_sampling import SMOTE

# Cargar el dataset
file_path = "smart_mobility_dataset.csv"
df = pd.read_csv(file_path)

# Copia del dataset para procesamiento
df_clean = df.copy()

#columna de tiempo
# a) Convertir la columna 'Timestamp' a tipo datetime
df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])

# b) Extraer una métrica de tiempo (minutos desde medianoche)
df_clean['time_in_minutes'] = df_clean['Timestamp'].dt.hour * 60 + df_clean['Timestamp'].dt.minute

# c) Normalizar esa métrica [0,1] usando MinMaxScaler
scaler_time = MinMaxScaler()
df_clean[['time_in_minutes']] = scaler_time.fit_transform(df_clean[['time_in_minutes']])

# d) (Opcional) Eliminar la columna original 'Timestamp'
df_clean.drop(columns=["Timestamp"], inplace=True)


df_clean.drop(columns=["Energy_Consumption_L_h"], inplace=True)
df_clean.drop(columns=["Emission_Levels_g_km"], inplace=True)


# Codificar variables categóricas con LabelEncoder
label_encoders = {}
categorical_columns = ["Traffic_Light_State", "Weather_Condition", "Traffic_Condition"]

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Separar características (X) y variable objetivo (y)
X = df_clean.drop(columns=["Traffic_Condition"])
y = df_clean["Traffic_Condition"]

print("Distribución de clases antes de SMOTE:")
print(y.value_counts())

# Crear el objeto SMOTE
smote = SMOTE(random_state=42)

# Ajustar y transformar
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Distribución de clases después de SMOTE:")
print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2, 
    random_state=42
)

# Definir la red neuronal
# model = keras.Sequential([
#     layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(16, activation='relu', kernel_regularizer= 'l2'),
#     layers.Dense(3, activation='softmax')  # 3 clases: High, Medium, Low
# ])
model = keras.Sequential([
    layers.Dense(32, activation='relu',
                 input_shape=(X_train.shape[1],),
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax',
                 kernel_regularizer=regularizers.l2(0.01))
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
epochs = 100
history = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_test, y_test), 
    epochs=epochs, 
    batch_size=32
)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Precisión en test: {test_acc:.4f}')

# Generar predicciones para el conjunto de prueba
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

# Reporte de clasificación
report = classification_report(y_test, y_pred)
print("Reporte de clasificación:")
print(report)

# Graficar la matriz de confusión
plt.figure(figsize=(8,6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap='Blues',
    xticklabels=label_encoders["Traffic_Condition"].classes_,
    yticklabels=label_encoders["Traffic_Condition"].classes_
)
plt.ylabel('Valores Reales')
plt.xlabel('Predicciones')
plt.title('Matriz de Confusión (SMOTE)')
plt.show()
