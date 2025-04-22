import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np



# Cargar el dataset desde Kaggle (o archivo local)
data = pd.read_csv('smart_mobility_dataset.csv')  

print("Primera fila del dataset:")
print(data.head())

print("Resumen de las columnas y sus tipos de datos:")
print(data.info())

print("Revisión de valores nulos:")
print(data.isnull().sum())


# Eliminar columnas irrelevantes
data.drop(['Emission_Levels_g_km', 'Energy_Consumption_L_h'], axis=1, inplace=True)

# Normalización de datos numéricos
scaler = StandardScaler()
data[['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']] = scaler.fit_transform(
    data[['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']]
)

weather_map = {
    'Clear': 0,
    'Rain': 1,
    'Snow': 2,
    'Fog': 3,
    'Storm': 4,
    'Cloudy': 5
}

# Aplicar la transformación
data['Weather_Condition'] = data['Weather_Condition'].map(weather_map)

traffic_light_map = {
    'Green': 0,
    'Yellow': 1,
    'Red': 2
}

data['Traffic_Light_State'] = data['Traffic_Light_State'].map(traffic_light_map)  


# Mapear la columna objetivo (Traffic_Condition) en tres clases
congestion_map = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
data['Congestion_Level'] = data['Traffic_Condition'].map(congestion_map)

data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Extraer características temporales útiles
data['Hour'] = data['Timestamp'].dt.hour
data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
data['Is_Weekend'] = (data['Day_of_Week'] >= 5).astype(int)

# Eliminar la columna original 'Timestamp' ya que no se requiere más
data.drop('Timestamp', axis=1, inplace=True)

# Verificar que las nuevas columnas estén correctamente agregadas
print(data[['Hour', 'Day_of_Week', 'Is_Weekend']].head())

# Eliminar la columna original de congestión
data.drop('Traffic_Condition', axis=1, inplace=True)

# Verificar que el dataset esté limpio
print(data.head())

# Dividir el dataset en conjuntos de entrenamiento y prueba
X = data.drop('Congestion_Level', axis=1)
y = data['Congestion_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Conjunto de entrenamiento:")
print(X_train_smote.shape, y_train_smote.shape)

print("Conjunto de prueba:")
print(X_test.shape, y_test.shape)

print("Tipos de datos por columna:")
print(data.dtypes)



model = keras.models.Sequential([
    layers.Input(shape=(X_train_smote.shape[1],)),
    layers.Dense(256, activation='leaky_relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train_smote, y_train_smote,
    validation_split=0.2,
    epochs=120,
    batch_size=16,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)]
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {test_loss:.4f}")
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Ajustar umbral para reducir errores en la clase 1
y_pred[(y_pred_proba[:, 1] >= 0.6) & (y_pred_proba[:, 2] < 0.7)] = 1
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()