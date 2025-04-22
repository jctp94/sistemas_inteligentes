import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import time
from sklearn.model_selection import StratifiedKFold
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

# Examinar distribución de clases
print("\nDistribución de clases (Traffic_Condition):")
print(df['Traffic_Condition'].value_counts())

# Eliminar columnas irrelevantes
df_clean = df.drop(columns=["Emission_Levels_g_km", "Energy_Consumption_L_h"])

# Convertir timestamp y extraer características temporales más completas
df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
df_clean['Hour'] = df_clean['Timestamp'].dt.hour
df_clean['Day_of_Week'] = df_clean['Timestamp'].dt.dayofweek
df_clean['Is_Weekend'] = (df_clean['Day_of_Week'] >= 5).astype(int)
df_clean['Is_Rush_Hour'] = ((df_clean['Hour'] >= 7) & (df_clean['Hour'] <= 9) | 
                           (df_clean['Hour'] >= 16) & (df_clean['Hour'] <= 19)).astype(int)

# Normalizar variables numéricas
numeric_cols = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 
                'Latitude', 'Longitude', 'Hour', 'Day_of_Week']
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
    print(f"\nMapeo de {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ==================================
# 3. Preparar características y etiquetas
# ==================================
X = df_clean.drop(columns=["Traffic_Condition"])
y = df_clean["Traffic_Condition"]

# Análisis de correlación
print("\nAnalizando correlaciones con la variable objetivo...")
correlations = df_clean.corr()['Traffic_Condition'].sort_values(ascending=False)
print(correlations)

# ==================================
# 4. Aplicar SMOTE para balancear clases
# ==================================
print("\nAplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=RANDOM_SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Distribución original de clases:")
print(pd.Series(y).value_counts())
print("Distribución después de SMOTE:")
print(pd.Series(y_resampled).value_counts())

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
print(f"Distribución de clases en entrenamiento: {np.unique(y_train, return_counts=True)}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}")
print(f"Distribución de clases en prueba: {np.unique(y_test, return_counts=True)}")

# ==================================
# 6. Definir el modelo de red neuronal
# ==================================
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), 
                    kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu',
                    kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
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
# 7. Implementación de Algoritmo Genético Mejorado
# ==================================
class ImprovedGeneticAlgorithm:
    def __init__(self, model, population_size=100, elite_ratio=0.1):
        self.model = model
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.elite_size = max(2, int(population_size * elite_ratio))
        
        # Obtener las formas y tamaños de los pesos del modelo
        self.weights_shapes = [w.shape for w in model.get_weights()]
        self.weights_sizes = [np.prod(s) for s in self.weights_shapes]
        self.total_weights = sum(self.weights_sizes)
        
        # Inicializar población con mayor diversidad
        self.population = self._initialize_diverse_population()
        
        # Mejor solución
        self.best_solution = None
        self.best_fitness = 0
        self.best_f1_score = 0
        self.fitness_history = []
        self.f1_history = []
        self.class_distribution_history = []
        
    def _initialize_diverse_population(self):
        """Inicializa población con mayor diversidad usando diferentes métodos"""
        population = []
        
        # 25% población random normal
        for _ in range(self.population_size // 4):
            population.append([np.random.normal(0, 0.1, size=shape) for shape in self.weights_shapes])
        
        # 25% población random uniforme
        for _ in range(self.population_size // 4):
            population.append([np.random.uniform(-0.1, 0.1, size=shape) for shape in self.weights_shapes])
        
        # 25% población basada en pesos iniciales del modelo
        initial_weights = model.get_weights()
        for _ in range(self.population_size // 4):
            # Perturbar los pesos iniciales
            perturbed_weights = []
            for w in initial_weights:
                perturbation = np.random.normal(0, 0.05, size=w.shape)
                perturbed_weights.append(w + perturbation)
            population.append(perturbed_weights)
        
        # 25% población mixta con diferentes escalas
        for _ in range(self.population_size - len(population)):
            if _ % 2 == 0:
                population.append([np.random.normal(0, 0.2, size=shape) for shape in self.weights_shapes])
            else:
                population.append([np.random.normal(0, 0.05, size=shape) for shape in self.weights_shapes])
        
        return population
    
    def evaluate_individual(self, weights, validation_data=None):
        """Evalúa un individuo usando múltiples métricas"""
        # Aplicar pesos al modelo
        self.model.set_weights(weights)
        
        # Usar conjunto de validación completo
        if validation_data is None:
            # Crear un conjunto de validación estratificado
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            train_idx, val_idx = next(skf.split(X_train, y_train))
            X_val, y_val = X_train[val_idx], y_train[val_idx]
        else:
            X_val, y_val = validation_data
        
        # Evaluar
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        
        # Calcular múltiples métricas
        accuracy = np.mean(y_pred == y_val)
        bal_accuracy = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        # Obtener distribución de clases en predicciones
        class_dist = np.bincount(y_pred, minlength=len(np.unique(y)))
        class_dist = class_dist / np.sum(class_dist)
        
        # Score compuesto (ponderando diferentes métricas)
        # Damos más peso al f1-score para contrarrestar el sesgo de clases
        composite_score = 0.2 * accuracy + 0.3 * bal_accuracy + 0.5 * f1
        
        return {
            'fitness': composite_score,
            'accuracy': accuracy,
            'balanced_accuracy': bal_accuracy,
            'f1_score': f1,
            'class_distribution': class_dist
        }
    
    def selection_tournament(self, fitness_scores, tournament_size=3):
        """Selección por torneo"""
        selected_indices = []
        
        # Asegurar que los elites siempre pasen
        elite_indices = np.argsort([f['fitness'] for f in fitness_scores])[-self.elite_size:]
        selected_indices.extend(elite_indices)
        
        # Completar la selección con torneos
        while len(selected_indices) < self.population_size:
            # Seleccionar individuos aleatorios para el torneo
            tournament_indices = np.random.choice(self.population_size, size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i]['fitness'] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        return selected_indices
    
    def crossover_uniform(self, parent1, parent2):
        """Implementa cruce uniforme entre padres"""
        child = []
        for w1, w2 in zip(parent1, parent2):
            # Máscara de cruce uniforme
            mask = np.random.random(w1.shape) < 0.5
            child_w = np.copy(w1)
            child_w[mask] = w2[mask]
            child.append(child_w)
        return child
    
    def mutate(self, individual, mutation_rate=0.1, mutation_scale=0.1):
        """Aplica mutación con tasa adaptativa"""
        mutated = []
        for w in individual:
            # Crear máscara de mutación
            mask = np.random.random(w.shape) < mutation_rate
            
            # Aplicar mutación solo a elementos seleccionados
            mutation = np.random.normal(0, mutation_scale, size=w.shape)
            mutation[~mask] = 0  # Poner a cero las posiciones que no mutan
            
            # Agregar la mutación a los pesos
            mutated_w = w + mutation
            mutated.append(mutated_w)
        
        return mutated
    
    def run(self, generations=50, validation_data=None):
        """Ejecuta el algoritmo genético por un número de generaciones"""
        print("\nIniciando algoritmo genético mejorado...")
        
        # Crear conjunto de validación fijo para toda la ejecución
        if validation_data is None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            train_idx, val_idx = next(skf.split(X_train, y_train))
            validation_data = (X_train[val_idx], y_train[val_idx])
        
        for generation in range(generations):
            start_time = time.time()
            
            # Evaluar población actual
            fitness_scores = [self.evaluate_individual(individual, validation_data) 
                             for individual in self.population]
            
            # Obtener mejor individuo de esta generación
            best_idx = np.argmax([f['fitness'] for f in fitness_scores])
            current_best = fitness_scores[best_idx]
            
            # Actualizar mejor solución global si es mejor
            if current_best['fitness'] > self.best_fitness:
                self.best_solution = self.population[best_idx]
                self.best_fitness = current_best['fitness']
                self.best_f1_score = current_best['f1_score']
            
            # Guardar historial
            self.fitness_history.append(current_best['fitness'])
            self.f1_history.append(current_best['f1_score'])
            
            # Guardar distribución de clases del mejor individuo
            self.class_distribution_history.append(current_best['class_distribution'])
            
            # Imprimir progreso con más métricas
            gen_time = time.time() - start_time
            print(f"Gen {generation+1}/{generations}: " +
                  f"Fitness = {current_best['fitness']:.4f}, " +
                  f"F1 = {current_best['f1_score']:.4f}, " +
                  f"Acc = {current_best['accuracy']:.4f}, " +
                  f"Class Dist = {np.round(current_best['class_distribution'], 2)}, " +
                  f"Time = {gen_time:.2f}s")
            
            # Crear nueva generación (excepto en la última iteración)
            if generation < generations - 1:
                # Selección - Elegir los mejores individuos por torneo
                selected_indices = self.selection_tournament(fitness_scores)
                new_population = [self.population[i] for i in selected_indices[:self.elite_size]]
                
                # Tasa de mutación adaptativa - decrece con el progreso
                mutation_rate = 0.2 * (1 - generation / generations)
                
                # Cruce y mutación para crear el resto de la población
                while len(new_population) < self.population_size:
                    # Seleccionar padres por torneo
                    parent1_idx = np.random.choice(selected_indices)
                    parent2_idx = np.random.choice(selected_indices)
                    
                    # Evitar seleccionar el mismo padre
                    while parent2_idx == parent1_idx:
                        parent2_idx = np.random.choice(selected_indices)
                    
                    # Cruce uniforme
                    child = self.crossover_uniform(
                        self.population[parent1_idx], 
                        self.population[parent2_idx]
                    )
                    
                    # Mutación adaptativa
                    child = self.mutate(child, mutation_rate=mutation_rate)
                    
                    new_population.append(child)
                
                self.population = new_population
        
        print(f"\nMejor fitness alcanzado: {self.best_fitness:.4f}")
        print(f"Mejor F1-score alcanzado: {self.best_f1_score:.4f}")
        
        # Aplicar la mejor solución encontrada al modelo
        self.model.set_weights(self.best_solution)
        return {
            'best_fitness': self.best_fitness,
            'best_f1_score': self.best_f1_score,
            'fitness_history': self.fitness_history,
            'f1_history': self.f1_history,
            'class_distribution_history': self.class_distribution_history
        }

# ==================================
# 8. Aplicar algoritmo genético mejorado
# ==================================
print("\nOptimizando con algoritmo genético mejorado...")
ga = ImprovedGeneticAlgorithm(model=model, population_size=100, elite_ratio=0.1)
ga_results = ga.run(generations=50)

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
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

# Evaluar en conjunto de prueba
trad_loss, trad_acc = model_traditional.evaluate(X_test, y_test)
print(f"Precisión en test (enfoque tradicional): {trad_acc:.4f}")

# ==================================
# 11. Reporte de clasificación y matriz de confusión para ambos modelos
# ==================================
# Obtener predicciones del modelo genético
y_pred_prob_ga = model.predict(X_test)
y_pred_ga = np.argmax(y_pred_prob_ga, axis=1)

# Obtener predicciones del modelo tradicional
y_pred_prob_trad = model_traditional.predict(X_test)
y_pred_trad = np.argmax(y_pred_prob_trad, axis=1)

# Matriz de confusión - Genético
cm_ga = confusion_matrix(y_test, y_pred_ga)
print("\nMatriz de confusión (modelo genético):")
print(cm_ga)

# Reporte detallado - Genético
print("\nReporte de clasificación (modelo genético):")
print(classification_report(y_test, y_pred_ga))

# Matriz de confusión - Tradicional
cm_trad = confusion_matrix(y_test, y_pred_trad)
print("\nMatriz de confusión (modelo tradicional):")
print(cm_trad)

# Reporte detallado - Tradicional
print("\nReporte de clasificación (modelo tradicional):")
print(classification_report(y_test, y_pred_trad))

# ==================================
# 12. Visualizaciones
# ==================================
plt.figure(figsize=(20, 15))

# 1. Matriz de confusión - Genético
plt.subplot(3, 2, 1)
class_names = label_encoders["Traffic_Condition"].classes_
sns.heatmap(cm_ga, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title("Matriz de Confusión (Genético)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")

# 2. Matriz de confusión - Tradicional
plt.subplot(3, 2, 2)
sns.heatmap(cm_trad, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title("Matriz de Confusión (Tradicional)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")

# 3. Evolución del fitness y F1-score
plt.subplot(3, 2, 3)
plt.plot(range(1, len(ga_results['fitness_history']) + 1), ga_results['fitness_history'], 'b-', linewidth=2, label='Fitness')
plt.plot(range(1, len(ga_results['f1_history']) + 1), ga_results['f1_history'], 'r--', linewidth=2, label='F1-Score')
plt.xlabel('Generación')
plt.ylabel('Puntuación')
plt.title('Evolución del Fitness y F1-Score Durante la Optimización')
plt.legend()
plt.grid(True)

# 4. Distribución de clases predichas por generación
plt.subplot(3, 2, 4)
class_dist_history = np.array(ga_results['class_distribution_history'])
for i in range(class_dist_history.shape[1]):
    plt.plot(range(1, len(ga_results['class_distribution_history']) + 1), 
             class_dist_history[:, i], 
             label=f'Clase {class_names[i]}')
plt.xlabel('Generación')
plt.ylabel('Proporción de predicciones')
plt.title('Evolución de la distribución de clases predichas')
plt.legend()
plt.grid(True)

# 5. Comparativa de métricas
plt.subplot(3, 2, 5)
# Obtener métricas detalladas para ambos modelos
report_ga = classification_report(y_test, y_pred_ga, output_dict=True)
report_trad = classification_report(y_test, y_pred_trad, output_dict=True)

# Extraer métricas F1 por clase y promedio
metrics = ['precision', 'recall', 'f1-score']
classes = [class_names[0], class_names[1], class_names[2], 'macro avg']
x = np.arange(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    ga_values = [report_ga[str(j)][metric] for j in range(3)] + [report_ga['macro avg'][metric]]
    trad_values = [report_trad[str(j)][metric] for j in range(3)] + [report_trad['macro avg'][metric]]
    
    plt.bar(x - width/2 + i*width/len(metrics), ga_values, width/len(metrics), label=f'GA {metric}', alpha=0.7)
    plt.bar(x + width/2 + i*width/len(metrics), trad_values, width/len(metrics), label=f'Trad {metric}', alpha=0.7)

plt.title('Comparación de Métricas por Clase')
plt.xticks(x, classes)
plt.ylabel('Puntuación')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

# 6. Historia de entrenamiento tradicional
plt.subplot(3, 2, 6)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Historia de Entrenamiento Tradicional')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("comparativa_metodos_mejorada.png")
plt.close()

print("\nLas visualizaciones se han guardado como 'comparativa_metodos_mejorada.png'")

# ==================================
# 13. Resumen y conclusiones
# ==================================
print("\n===== RESUMEN DE RESULTADOS =====")
print(f"Precisión con algoritmo genético mejorado: {test_acc:.4f}")
print(f"Precisión con entrenamiento tradicional: {trad_acc:.4f}")
print(f"Diferencia: {test_acc - trad_acc:.4f}")

# Calcular F1-scores
f1_ga = f1_score(y_test, y_pred_ga, average='macro')
f1_trad = f1_score(y_test, y_pred_trad, average='macro')

print(f"F1-Score (macro) con algoritmo genético mejorado: {f1_ga:.4f}")
print(f"F1-Score (macro) con entrenamiento tradicional: {f1_trad:.4f}")
print(f"Diferencia en F1-Score: {f1_ga - f1_trad:.4f}")

print("\n===== ANÁLISIS DEL SESGO =====")
print("Distribución de clases en predicciones (Genético):")
print(np.bincount(y_pred_ga, minlength=len(np.unique(y))))

print("Distribución de clases en predicciones (Tradicional):")
print(np.bincount(y_pred_trad, minlength=len(np.unique(y))))

if test_acc > trad_acc and f1_ga > f1_trad:
    print("\nCONCLUSIÓN: El algoritmo genético mejorado superó al entrenamiento tradicional tanto en precisión como en F1-Score.")
elif test_acc > trad_acc:
    print("\nCONCLUSIÓN: El algoritmo genético mejorado superó al entrenamiento tradicional en precisión, pero no en F1-Score.")
elif f1_ga > f1_trad:
    print("\nCONCLUSIÓN: El algoritmo genético mejorado superó al entrenamiento tradicional en F1-Score, pero no en precisión.")
else:
    print("\nCONCLUSIÓN: El entrenamiento tradicional obtuvo mejores resultados que el algoritmo genético mejorado.")

print("\nProceso completo ejecutado correctamente.")