import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import time
warnings.filterwarnings('ignore')

# Semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==================================
# 1. Cargar y preprocesar el dataset
# ==================================
print("Cargando y preprocesando el dataset...")
file_path = "smart_mobility_dataset.csv"
df = pd.read_csv(file_path)

# Eliminar columnas irrelevantes
df_clean = df.drop(columns=["Emission_Levels_g_km", "Energy_Consumption_L_h"])

# Convertir timestamp y extraer minutos desde medianoche
df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
df_clean['time_in_minutes'] = df_clean['Timestamp'].dt.hour * 60 + df_clean['Timestamp'].dt.minute

# Normalizar minutos
scaler_time = MinMaxScaler()
df_clean[['time_in_minutes']] = scaler_time.fit_transform(df_clean[['time_in_minutes']])

# Normalizar variables numéricas
numeric_cols = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']
scaler_numeric = MinMaxScaler()
df_clean[numeric_cols] = scaler_numeric.fit_transform(df_clean[numeric_cols])

# Eliminar timestamp original después de extraer características
df_clean.drop(columns=["Timestamp"], inplace=True)

# ==================================
# 2. Codificar variables categóricas
# ==================================
label_encoders = {}
categorical_columns = ["Traffic_Light_State", "Weather_Condition", "Traffic_Condition"]

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# ==================================
# 3. Preparar características y etiquetas
# ==================================
X = df_clean.drop(columns=["Traffic_Condition"])
y = df_clean["Traffic_Condition"]

# ==================================
# 4. Aplicar SMOTE para balancear clases
# ==================================
print("\nAplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=RANDOM_SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ==================================
# 5. Convertir a numpy arrays y dividir datos
# ==================================
# Importante: Convertir a numpy arrays para evitar problemas de indexación con pandas
X_resampled_np = X_resampled.values
y_resampled_np = y_resampled.values

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_np, y_resampled_np, test_size=0.2, random_state=RANDOM_SEED, stratify=y_resampled_np
)

print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}")

# ==================================
# 6. Definir el modelo de red neuronal
# ==================================
def create_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],), 
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(len(np.unique(y)), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Crear el modelo base
model = create_model()
model.summary()

# ==================================
# 7. Implementación de Algoritmo Genético Simplificado
# ==================================
class SimpleGeneticAlgorithm:
    def __init__(self, model, population_size=20):
        self.model = model
        self.population_size = population_size
        
        # Obtener las formas y tamaños de los pesos del modelo
        self.weights_shapes = [w.shape for w in model.get_weights()]
        self.weights_sizes = [np.prod(s) for s in self.weights_shapes]
        self.total_weights = sum(self.weights_sizes)
        
        # Inicializar población
        self.population = [self.generate_random_weights() for _ in range(population_size)]
        
        # Mejor solución
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []
        
    def generate_random_weights(self):
        """Genera un conjunto aleatorio de pesos para el modelo"""
        return [np.random.normal(0, 0.1, size=shape) for shape in self.weights_shapes]
    
    def evaluate_individual(self, weights, validation_size=1000):
        """Evalúa un individuo con un subconjunto de datos para acelerar el proceso"""
        # Aplicar pesos al modelo
        self.model.set_weights(weights)
        
        # Usar un subconjunto de datos para evaluación rápida
        indices = np.random.choice(len(X_train), size=min(validation_size, len(X_train)), replace=False)
        X_val = X_train[indices]
        y_val = y_train[indices]
        
        # Evaluar
        _, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        return accuracy
    
    def run(self, generations=10):
        """Ejecuta el algoritmo genético por un número de generaciones"""
        print("\nIniciando algoritmo genético simplificado...")
        
        for generation in range(generations):
            start_time = time.time()
            
            # Evaluar población actual
            fitness_scores = [self.evaluate_individual(individual) for individual in self.population]
            
            # Actualizar mejor solución
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_solution = self.population[best_idx]
                self.best_fitness = fitness_scores[best_idx]
            
            self.fitness_history.append(self.best_fitness)
            
            # Imprimir progreso
            gen_time = time.time() - start_time
            print(f"Generación {generation+1}/{generations}: " +
                  f"Mejor fitness = {self.best_fitness:.4f}, " +
                  f"Tiempo = {gen_time:.2f}s")
            
            # Crear nueva generación (excepto en la última iteración)
            if generation < generations - 1:
                # Selección - Elegir los mejores individuos
                elite_count = max(2, self.population_size // 10)
                elite_indices = np.argsort(fitness_scores)[-elite_count:]
                
                new_population = [self.population[i] for i in elite_indices]
                
                # Cruce y mutación para crear el resto de la población
                while len(new_population) < self.population_size:
                    # Seleccionar padres (método simple: torneo entre 3 individuos)
                    parent_indices = np.random.choice(self.population_size, size=3, replace=False)
                    parent_fitness = [fitness_scores[i] for i in parent_indices]
                    parent1_idx = parent_indices[np.argmax(parent_fitness)]
                    
                    parent_indices = np.random.choice(self.population_size, size=3, replace=False)
                    parent_fitness = [fitness_scores[i] for i in parent_indices]
                    parent2_idx = parent_indices[np.argmax(parent_fitness)]
                    
                    # Crear hijo mediante cruce simple y mutación
                    child = []
                    for w1, w2 in zip(self.population[parent1_idx], self.population[parent2_idx]):
                        # Cruce (50% de cada padre)
                        if np.random.random() < 0.5:
                            child_w = w1.copy()
                        else:
                            child_w = w2.copy()
                        
                        # Mutación (solo algunos genes)
                        mask = np.random.random(child_w.shape) < 0.1  # 10% de probabilidad
                        child_w[mask] += np.random.normal(0, 0.1, size=np.sum(mask))
                        
                        child.append(child_w)
                    
                    new_population.append(child)
                
                self.population = new_population
        
        # Aplicar la mejor solución encontrada al modelo
        self.model.set_weights(self.best_solution)
        return self.best_fitness, self.fitness_history

# ==================================
# 8. Aplicar algoritmo genético simplificado
# ==================================
print("\nOptimizando pesos iniciales con algoritmo genético...")
ga = SimpleGeneticAlgorithm(model=model, population_size=20)
best_fitness, fitness_history = ga.run(generations=10)

# ==================================
# 9. Evaluar el modelo en conjunto de prueba
# ==================================
print("\nEvaluando modelo con pesos optimizados en conjunto de prueba...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en test: {test_acc:.4f}")
print(f"Pérdida en test: {test_loss:.4f}")

# ==================================
# 10. Comparación con entrenamiento tradicional
# ==================================
print("\nEntrenando modelo con enfoque tradicional para comparación...")

# Crear modelo fresco para comparación
model_traditional = create_model()

# Entrenar con enfoque tradicional
history = model_traditional.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluar en conjunto de prueba
trad_loss, trad_acc = model_traditional.evaluate(X_test, y_test)
print(f"Precisión en test (enfoque tradicional): {trad_acc:.4f}")

# ==================================
# 11. Reporte de clasificación y matriz de confusión
# ==================================
# Obtener predicciones del modelo genético
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión (modelo genético):")
print(cm)

# Reporte detallado
print("\nReporte de clasificación (modelo genético):")
print(classification_report(y_test, y_pred))

# ==================================
# 12. Visualizaciones
# ==================================
plt.figure(figsize=(15, 10))

# 1. Matriz de confusión
plt.subplot(2, 2, 1)
class_names = label_encoders["Traffic_Condition"].classes_
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title("Matriz de Confusión (Genético)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")

# 2. Evolución de fitness
plt.subplot(2, 2, 2)
plt.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', linewidth=2)
plt.xlabel('Generación')
plt.ylabel('Fitness (Accuracy)')
plt.title('Evolución del Fitness Durante la Optimización')
plt.grid(True)

# 3. Comparativa de accuracy
plt.subplot(2, 2, 3)
plt.bar(['Genético', 'Tradicional'], [test_acc, trad_acc], color=['blue', 'green'])
plt.ylim([0, 1])
plt.title('Comparación de Precisión en Test')
plt.ylabel('Accuracy')

# 4. Historia de entrenamiento tradicional
plt.subplot(2, 2, 4)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Historia de Entrenamiento Tradicional')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("comparativa_metodos.png")
plt.close()

print("\nLas visualizaciones se han guardado como 'comparativa_metodos.png'")

# ==================================
# 13. Resumen y conclusiones
# ==================================
print("\n===== RESUMEN DE RESULTADOS =====")
print(f"Precisión con algoritmo genético: {test_acc:.4f}")
print(f"Precisión con entrenamiento tradicional: {trad_acc:.4f}")
print(f"Diferencia: {test_acc - trad_acc:.4f}")

if test_acc > trad_acc:
    print("\nEl algoritmo genético mejoró la inicialización de pesos.")
else:
    print("\nEl entrenamiento tradicional obtuvo mejores resultados.")

print("\nProceso completo ejecutado correctamente.")