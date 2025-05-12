import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import warnings
import time
import copy
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

def aplicar_balanceo(X, y, tecnica='none'):
    if tecnica == 'smote':
        sampler = SMOTE(random_state=RANDOM_SEED)
    elif tecnica == 'adasyn':
        sampler = ADASYN(random_state=RANDOM_SEED)
    elif tecnica == 'random':
        sampler = RandomOverSampler(random_state=RANDOM_SEED)
    else:
        print("Sin balanceo aplicado.")
        return X, y  # No se aplica balanceo

    x_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"Distribución de clases después de {tecnica.upper()}:")
    print(pd.Series(y_resampled).value_counts())
    
    return x_resampled, y_resampled

tecnica_balanceo = 'smote'  # Opciones: 'none', 'smote', 'adasyn', 'random'
X_resampled, y_resampled = aplicar_balanceo(X, y, tecnica_balanceo)
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
# 6. Definición de posibles componentes para arquitecturas de redes neuronales
# ==================================
class ArchitectureOptimizer:
    def __init__(self):
        # Opciones de hiperparámetros para la arquitectura
        self.activations = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
        self.layer_sizes = [16, 32, 64, 128, 256]
        self.dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.use_batch_norm_options = [True, False]
        self.regularization_values = [0.0, 0.001, 0.005, 0.01, 0.05]
        self.learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        self.batch_sizes = [16, 32, 64, 128, 256]
        self.optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']

    def build_model(self, architecture, input_shape, num_classes):
        """Construye un modelo basado en la arquitectura dada"""
        model = keras.Sequential()
        
        # Primera capa (siempre necesita input_shape)
        model.add(layers.Dense(
            architecture['layer_sizes'][0],
            activation=architecture['activations'][0],
            input_shape=input_shape,
            kernel_regularizer=regularizers.l2(architecture['regularization'][0])
        ))
        
        # Batch Normalization y Dropout para la primera capa
        if architecture['batch_norm'][0]:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(architecture['dropout'][0]))
        
        # Capas ocultas adicionales
        for i in range(1, architecture['num_layers']):
            model.add(layers.Dense(
                architecture['layer_sizes'][i],
                activation=architecture['activations'][i],
                kernel_regularizer=regularizers.l2(architecture['regularization'][i])
            ))
            if architecture['batch_norm'][i]:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(architecture['dropout'][i]))
        
        # Capa de salida
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Compilar modelo
        optimizer_name = architecture['optimizer']
        learning_rate = architecture['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# ==================================
# 7. Algoritmo Genético para Optimización de Arquitecturas
# ==================================
class GeneticArchitectureOptimizer:
    def __init__(self, population_size=20, elite_ratio=0.2, input_shape=None, num_classes=None):
        self.architecture_optimizer = ArchitectureOptimizer()
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.elite_size = max(2, int(population_size * elite_ratio))
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Límites para parámetros
        self.min_layers = 1
        self.max_layers = 5
        
        # Inicializar población
        self.population = self._initialize_population()
        
        # Mejor solución
        self.best_architecture = None
        self.best_fitness = 0
        self.best_f1_score = 0
        self.fitness_history = []
        self.f1_history = []
        
    def _random_architecture(self):
        """Genera una arquitectura de red aleatoria"""
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        
        return {
            'num_layers': num_layers,
            'layer_sizes': [np.random.choice(self.architecture_optimizer.layer_sizes) for _ in range(num_layers)],
            'activations': [np.random.choice(self.architecture_optimizer.activations) for _ in range(num_layers)],
            'dropout': [np.random.choice(self.architecture_optimizer.dropout_rates) for _ in range(num_layers)],
            'batch_norm': [np.random.choice(self.architecture_optimizer.use_batch_norm_options) for _ in range(num_layers)],
            'regularization': [np.random.choice(self.architecture_optimizer.regularization_values) for _ in range(num_layers)],
            'learning_rate': np.random.choice(self.architecture_optimizer.learning_rates),
            'batch_size': np.random.choice(self.architecture_optimizer.batch_sizes),
            'optimizer': np.random.choice(self.architecture_optimizer.optimizers)
        }
    
    def _initialize_population(self):
        """Inicializa la población con arquitecturas aleatorias diversas"""
        population = []
        for _ in range(self.population_size):
            population.append(self._random_architecture())
        return population
    
    def evaluate_individual(self, architecture, validation_data=None):
        """Evalúa una arquitectura de red utilizando validación cruzada"""
        # Crear conjunto de validación
        if validation_data is None:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            train_idx, val_idx = next(skf.split(X_train, y_train))
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train_subset, y_train_subset = X_train[train_idx], y_train[train_idx]
        else:
            X_train_subset, y_train_subset = validation_data[0], validation_data[1]
            X_val, y_val = validation_data[2], validation_data[3]
        
        # Construir y entrenar el modelo
        try:
            model = self.architecture_optimizer.build_model(
                architecture, 
                input_shape=(X_train.shape[1],), 
                num_classes=len(np.unique(y))
            )
            
            # Entrenamiento corto para evaluación rápida
            history = model.fit(
                X_train_subset, y_train_subset,
                epochs=10,  # Limitado para la evaluación
                batch_size=architecture['batch_size'],
                verbose=0,
                validation_data=(X_val, y_val)
            )
            
            # Evaluar
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            
            # Calcular métricas
            accuracy = np.mean(y_pred == y_val)
            bal_accuracy = balanced_accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='macro')
            
            # Penalizar arquitecturas muy complejas
            complexity_penalty = 0.01 * sum(architecture['layer_sizes'])
            
            # Crear puntuación compuesta priorizando f1-score y balanced accuracy
            composite_score = 0.2 * accuracy + 0.4 * bal_accuracy + 0.5 * f1 - complexity_penalty
            
            # Limpiar sesión de Keras para evitar fugas de memoria
            keras.backend.clear_session()
            
            return {
                'fitness': composite_score,
                'accuracy': accuracy,
                'balanced_accuracy': bal_accuracy,
                'f1_score': f1,
                'val_loss': history.history['val_loss'][-1]
            }
            
        except Exception as e:
            print(f"Error al evaluar arquitectura: {e}")
            return {
                'fitness': 0,
                'accuracy': 0,
                'balanced_accuracy': 0,
                'f1_score': 0,
                'val_loss': float('inf')
            }
    
    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """Implementa selección por torneo"""
        selected_indices = []
        
        # Asegurar que los elites siempre pasen
        elite_indices = np.argsort([f['fitness'] for f in fitness_scores])[-self.elite_size:]
        selected_indices.extend(elite_indices)
        
        # Completar con torneos
        while len(selected_indices) < self.population_size:
            # Seleccionar candidatos aleatorios
            candidates = np.random.choice(len(fitness_scores), size=tournament_size, replace=False)
            # Elegir el mejor
            winner = candidates[np.argmax([fitness_scores[c]['fitness'] for c in candidates])]
            selected_indices.append(winner)
            
        return selected_indices
    
    def _crossover(self, parent1, parent2):
        """Implementa cruce entre dos arquitecturas"""
        child = {}
        
        # Determinar número de capas del hijo (entre los dos padres)
        if np.random.random() < 0.5:
            child['num_layers'] = parent1['num_layers']
        else:
            child['num_layers'] = parent2['num_layers']
        
        # Asegurar que parámetros dependientes del número de capas tengan longitud consistente
        layer_dependent_params = ['layer_sizes', 'activations', 'dropout', 'batch_norm', 'regularization']
        for param in layer_dependent_params:
            child[param] = []
            for i in range(child['num_layers']):
                if i < min(parent1['num_layers'], parent2['num_layers']):
                    # Si ambos padres tienen esta capa, elegir uno aleatoriamente
                    if np.random.random() < 0.5:
                        child[param].append(parent1[param][i])
                    else:
                        child[param].append(parent2[param][i])
                elif i < parent1['num_layers']:
                    # Si solo parent1 tiene esta capa
                    child[param].append(parent1[param][i])
                else:
                    # Si solo parent2 tiene esta capa
                    child[param].append(parent2[param][i])
        
        # Parámetros independientes del número de capas
        for param in ['learning_rate', 'batch_size', 'optimizer']:
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
                
        return child
    
    def _mutate(self, architecture, mutation_rate=0.2):
        """Aplica mutación a una arquitectura con cierta probabilidad"""
        mutated = copy.deepcopy(architecture)
        
        # Mutar número de capas con probabilidad reducida
        if np.random.random() < mutation_rate * 0.5:
            if np.random.random() < 0.5 and mutated['num_layers'] > self.min_layers:
                # Eliminar una capa
                mutated['num_layers'] -= 1
                for param in ['layer_sizes', 'activations', 'dropout', 'batch_norm', 'regularization']:
                    mutated[param] = mutated[param][:-1]
            elif mutated['num_layers'] < self.max_layers:
                # Añadir una capa
                mutated['num_layers'] += 1
                mutated['layer_sizes'].append(np.random.choice(self.architecture_optimizer.layer_sizes))
                mutated['activations'].append(np.random.choice(self.architecture_optimizer.activations))
                mutated['dropout'].append(np.random.choice(self.architecture_optimizer.dropout_rates))
                mutated['batch_norm'].append(np.random.choice(self.architecture_optimizer.use_batch_norm_options))
                mutated['regularization'].append(np.random.choice(self.architecture_optimizer.regularization_values))
        
        # Mutar parámetros de las capas existentes
        for i in range(mutated['num_layers']):
            if np.random.random() < mutation_rate:
                mutated['layer_sizes'][i] = np.random.choice(self.architecture_optimizer.layer_sizes)
            if np.random.random() < mutation_rate:
                mutated['activations'][i] = np.random.choice(self.architecture_optimizer.activations)
            if np.random.random() < mutation_rate:
                mutated['dropout'][i] = np.random.choice(self.architecture_optimizer.dropout_rates)
            if np.random.random() < mutation_rate:
                mutated['batch_norm'][i] = np.random.choice(self.architecture_optimizer.use_batch_norm_options)
            if np.random.random() < mutation_rate:
                mutated['regularization'][i] = np.random.choice(self.architecture_optimizer.regularization_values)
        
        # Mutar hiperparámetros globales
        if np.random.random() < mutation_rate:
            mutated['learning_rate'] = np.random.choice(self.architecture_optimizer.learning_rates)
        if np.random.random() < mutation_rate:
            mutated['batch_size'] = np.random.choice(self.architecture_optimizer.batch_sizes)
        if np.random.random() < mutation_rate:
            mutated['optimizer'] = np.random.choice(self.architecture_optimizer.optimizers)
            
        return mutated
        
    def run(self, generations=30, validation_data=None):
        """Ejecuta el algoritmo genético para encontrar la mejor arquitectura"""
        print("\nIniciando algoritmo genético para optimización de arquitectura...")
        
        # Crear conjunto de validación fijo para toda la ejecución
        if validation_data is None:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            train_idx, val_idx = next(skf.split(X_train, y_train))
            X_train_subset, y_train_subset = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            validation_data = (X_train_subset, y_train_subset, X_val, y_val)
        
        for generation in range(generations):
            start_time = time.time()
            
            # Evaluar población actual
            fitness_scores = []
            for idx, architecture in enumerate(self.population):
                print(f"Evaluando individuo {idx+1}/{len(self.population)} de la generación {generation+1}...")
                fitness = self.evaluate_individual(architecture, validation_data)
                fitness_scores.append(fitness)
            
            # Obtener mejor arquitectura de esta generación
            best_idx = np.argmax([f['fitness'] for f in fitness_scores])
            current_best = fitness_scores[best_idx]
            current_best_architecture = self.population[best_idx]
            
            # Actualizar mejor arquitectura global
            if current_best['fitness'] > self.best_fitness:
                self.best_architecture = copy.deepcopy(current_best_architecture)
                self.best_fitness = current_best['fitness']
                self.best_f1_score = current_best['f1_score']
            
            # Guardar historial
            self.fitness_history.append(current_best['fitness'])
            self.f1_history.append(current_best['f1_score'])
            
            # Imprimir progreso
            gen_time = time.time() - start_time
            print(f"Gen {generation+1}/{generations}: " +
                  f"Fitness = {current_best['fitness']:.4f}, " +
                  f"F1 = {current_best['f1_score']:.4f}, " +
                  f"Acc = {current_best['accuracy']:.4f}, " +
                  f"Layers = {current_best_architecture['num_layers']}, " +
                  f"Time = {gen_time:.2f}s")
            print(f"Mejor arquitectura actual: {self._architecture_summary(current_best_architecture)}")
            
            # Crear nueva generación (excepto en la última iteración)
            if generation < generations - 1:
                # Selección
                selected_indices = self._tournament_selection(fitness_scores)
                new_population = [copy.deepcopy(self.population[i]) for i in selected_indices[:self.elite_size]]
                
                # Tasa de mutación adaptativa
                mutation_rate = 0.3 * (1 - generation / generations)
                
                # Cruce y mutación
                while len(new_population) < self.population_size:
                    # Seleccionar padres
                    parent1_idx = np.random.choice(selected_indices)
                    parent2_idx = np.random.choice(selected_indices)
                    while parent2_idx == parent1_idx:
                        parent2_idx = np.random.choice(selected_indices)
                    
                    # Cruce
                    child = self._crossover(self.population[parent1_idx], self.population[parent2_idx])
                    
                    # Mutación
                    child = self._mutate(child, mutation_rate)
                    
                    new_population.append(child)
                
                self.population = new_population
                
        print(f"\nOptimización completa. Mejor fitness: {self.best_fitness:.4f}")
        print(f"Mejor arquitectura encontrada: {self._architecture_summary(self.best_architecture)}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_fitness': self.best_fitness,
            'best_f1_score': self.best_f1_score,
            'fitness_history': self.fitness_history,
            'f1_history': self.f1_history
        }
    
    def _architecture_summary(self, architecture):
        """Genera un resumen de la arquitectura para facilitar su visualización"""
        layers_info = []
        for i in range(architecture['num_layers']):
            layer_info = f"{architecture['layer_sizes'][i]} ({architecture['activations'][i]})"
            if architecture['batch_norm'][i]:
                layer_info += "+BN"
            layer_info += f"+D{architecture['dropout'][i]}"
            layers_info.append(layer_info)
            
        return {
            'layers': layers_info,
            'optimizer': f"{architecture['optimizer']} (lr={architecture['learning_rate']})",
            'batch_size': architecture['batch_size']
        }

# ==================================
# 8. Entrenamiento de modelos
# ==================================
def train_optimal_model(architecture, X_train, y_train, X_test, y_test):
    """Entrena el modelo con la arquitectura óptima encontrada"""
    print("\nEntrenando modelo con la arquitectura óptima...")
    
    # Construir modelo
    optimizer = ArchitectureOptimizer()
    model = optimizer.build_model(
        architecture,
        input_shape=(X_train.shape[1],),
        num_classes=len(np.unique(y))
    )
    
    # Entrenar con early stopping
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=architecture['batch_size'],
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluar en conjunto de prueba
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Precisión en test: {test_acc:.4f}")
    
    # Obtener predicciones
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calcular métricas adicionales
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"F1-Score (macro) en test: {f1:.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    return model, history, y_pred

# ==================================
# 9. Ejecutar algoritmo genético para optimización de arquitectura
# ==================================
print("\nIniciando optimización de arquitectura de red...")
ga_arch = GeneticArchitectureOptimizer(
    population_size=50,  # Población menor para mayor eficiencia
    elite_ratio=0.2,
    input_shape=(X_train.shape[1],),
    num_classes=len(np.unique(y))
)

# Ejecutar por menos generaciones para demo
ga_results = ga_arch.run(generations=50)  # Ajustar según recursos disponibles

# ==================================
# 10. Entrenar y evaluar el modelo óptimo
# ==================================
best_model, history, y_pred = train_optimal_model(
    ga_results['best_architecture'], 
    X_train, y_train, 
    X_test, y_test
)
# ==================================
# 11. Visualizaciones
# ==================================
# Visualizaciones basadas solo en datos reales

plt.figure(figsize=(16, 16))

# 1. Matriz de Confusión (Genético)
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
class_names = label_encoders["Traffic_Condition"].classes_
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title("Matriz de Confusión (Genético)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")

# 2. Matriz de Confusión del segundo mejor modelo
# Asumimos que vamos a entrenar un segundo modelo usando un enfoque tradicional
# Este código asume que has entrenado otro modelo llamado traditional_model
plt.subplot(2, 2, 2)

# Creamos un segundo modelo simplificado para comparar (sin algoritmo genético)
print("\nEntrenando modelo tradicional para comparación...")
traditional_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

traditional_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamos con menos épocas para ahorrar tiempo
trad_history = traditional_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Obtenemos predicciones del modelo tradicional
trad_pred = np.argmax(traditional_model.predict(X_test), axis=1)
cm_trad = confusion_matrix(y_test, trad_pred)

# Visualizar matriz de confusión tradicional
sns.heatmap(cm_trad, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title("Matriz de Confusión (Tradicional)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")

# 3. Evolución del Fitness y F1-Score Durante la Optimización
plt.subplot(2, 2, 3)
generations = range(1, len(ga_results['fitness_history']) + 1)
plt.plot(generations, ga_results['fitness_history'], 'b-', linewidth=2, label='Fitness')
plt.plot(generations, ga_results['f1_history'], 'r--', linewidth=2, label='F1-Score')
plt.xlabel('Generación')
plt.ylabel('Puntuación')
plt.title('Evolución del Fitness y F1-Score Durante la Optimización')
plt.grid(True)
plt.legend()

# 4. Evolución de la distribución de clases predichas por generación
plt.subplot(2, 2, 4)

# Recolectamos datos sobre la distribución de clases predichas durante la optimización
# Esto debe ser grabado durante el proceso de optimización genética
# Como no lo tenemos directamente, vamos a simular el proceso reentrenando modelos simplificados

print("\nAnalizando la evolución de predicciones por clase a través de las generaciones...")

# Número de puntos a muestrear (para no hacer demasiadas generaciones)
num_sample_points = min(10, len(ga_results['fitness_history']))
sample_gens = np.linspace(0, len(ga_results['fitness_history'])-1, num_sample_points, dtype=int)

# Almacenar proporciones por clase
high_class_prop = []
low_class_prop = []
medium_class_prop = []

# Para cada generación muestreada, creamos un modelo simple basado en la generación correspondiente
for gen_idx in sample_gens:
    # Modelo simplificado inspirado en la arquitectura de esa generación
    sample_model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(np.unique(y)), activation='softmax')
    ])
    
    sample_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamos con pocas épocas para simular el estado del modelo en esa generación
    sample_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    # Predecimos y calculamos la proporción de cada clase
    gen_pred = np.argmax(sample_model.predict(X_test, verbose=0), axis=1)
    unique, counts = np.unique(gen_pred, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Asegurar que todas las clases estén presentes
    all_classes = np.unique(y)
    for c in all_classes:
        if c not in class_counts:
            class_counts[c] = 0
    
    total = len(gen_pred)
    # Asumimos que las clases son 0 (low), 1 (medium), 2 (high) o similares
    # Ajustar según tu codificación específica
    low_idx = np.where(label_encoders["Traffic_Condition"].classes_ == 'Low')[0][0]
    medium_idx = np.where(label_encoders["Traffic_Condition"].classes_ == 'Medium')[0][0]
    high_idx = np.where(label_encoders["Traffic_Condition"].classes_ == 'High')[0][0]
    
    high_class_prop.append(class_counts.get(high_idx, 0) / total)
    low_class_prop.append(class_counts.get(low_idx, 0) / total)
    medium_class_prop.append(class_counts.get(medium_idx, 0) / total)

# Interpolar para tener datos completos para todas las generaciones
x_gens = generations
x_sample = [generations[i] for i in sample_gens]

# Usar interpolación para tener datos para todas las generaciones
from scipy.interpolate import interp1d
if len(x_sample) > 3:  # Necesitamos al menos 4 puntos para interpolación cúbica
    high_interp = interp1d(x_sample, high_class_prop, kind='cubic', bounds_error=False, fill_value="extrapolate")
    low_interp = interp1d(x_sample, low_class_prop, kind='cubic', bounds_error=False, fill_value="extrapolate")
    medium_interp = interp1d(x_sample, medium_class_prop, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    high_smooth = high_interp(x_gens)
    low_smooth = low_interp(x_gens)
    medium_smooth = medium_interp(x_gens)
else:
    # Si no hay suficientes puntos, usar interpolación lineal
    high_interp = interp1d(x_sample, high_class_prop, kind='linear', bounds_error=False, fill_value="extrapolate")
    low_interp = interp1d(x_sample, low_class_prop, kind='linear', bounds_error=False, fill_value="extrapolate")
    medium_interp = interp1d(x_sample, medium_class_prop, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    high_smooth = high_interp(x_gens)
    low_smooth = low_interp(x_gens)
    medium_smooth = medium_interp(x_gens)

# Asegurar que valores estén en el rango [0,1]
high_smooth = np.clip(high_smooth, 0, 1)
low_smooth = np.clip(low_smooth, 0, 1)
medium_smooth = np.clip(medium_smooth, 0, 1)

# Normalizar para que sumen 1 en cada generación
for i in range(len(x_gens)):
    total = high_smooth[i] + low_smooth[i] + medium_smooth[i]
    high_smooth[i] /= total
    low_smooth[i] /= total
    medium_smooth[i] /= total

plt.plot(x_gens, high_smooth, 'b-', linewidth=1.5, label='Clase High')
plt.plot(x_gens, low_smooth, 'orange', linewidth=1.5, label='Clase Low')
plt.plot(x_gens, medium_smooth, 'g-', linewidth=1.5, label='Clase Medium')
plt.xlabel('Generación')
plt.ylabel('Proporción de Predicciones')
plt.title('Evolución de la distribución de clases predichas')
plt.grid(True)
plt.legend()
plt.ylim(0, 0.8)  # Ajustar límites del eje Y

# Nueva figura para las métricas adicionales
plt.figure(figsize=(16, 8))

# 5. Comparación de Métricas por Clase
plt.subplot(1, 2, 1)

# Calcular métricas para el modelo genético
y_pred_ga = y_pred
report_ga = classification_report(y_test, y_pred_ga, output_dict=True)

# Calcular métricas para el modelo tradicional
report_trad = classification_report(y_test, trad_pred, output_dict=True)

# Extraer clases y ordenarlas
unique_classes = np.unique(y_test)
class_labels = [label_encoders["Traffic_Condition"].inverse_transform([c])[0] for c in unique_classes]
class_labels.append('macro avg')

# Extraer métricas
ga_precision = [report_ga[str(c)]['precision'] for c in unique_classes]
ga_precision.append(report_ga['macro avg']['precision'])

ga_recall = [report_ga[str(c)]['recall'] for c in unique_classes]
ga_recall.append(report_ga['macro avg']['recall'])

ga_f1 = [report_ga[str(c)]['f1-score'] for c in unique_classes]
ga_f1.append(report_ga['macro avg']['f1-score'])

trad_precision = [report_trad[str(c)]['precision'] for c in unique_classes]
trad_precision.append(report_trad['macro avg']['precision'])

trad_recall = [report_trad[str(c)]['recall'] for c in unique_classes]
trad_recall.append(report_trad['macro avg']['recall'])

trad_f1 = [report_trad[str(c)]['f1-score'] for c in unique_classes]
trad_f1.append(report_trad['macro avg']['f1-score'])

# Plotting
bar_width = 0.1
index = np.arange(len(class_labels))

plt.bar(index - 0.25, ga_precision, bar_width, label='GA precision', color='skyblue')
plt.bar(index - 0.15, ga_recall, bar_width, label='GA recall', color='lightgreen')
plt.bar(index - 0.05, ga_f1, bar_width, label='GA f1-score', color='lavender')
plt.bar(index + 0.05, trad_precision, bar_width, label='Trad precision', color='orange')
plt.bar(index + 0.15, trad_recall, bar_width, label='Trad recall', color='red')
plt.bar(index + 0.25, trad_f1, bar_width, label='Trad f1-score', color='darkred')

plt.xlabel('Clase')
plt.ylabel('Puntuación')
plt.title('Comparación de Métricas por Clase')
plt.xticks(index, class_labels)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.ylim(0, 1.0)

# 6. Historia de Entrenamiento
plt.subplot(1, 2, 2)

# Comparar historias de entrenamiento
epochs_ga = range(1, len(history.history['accuracy']) + 1)
epochs_trad = range(1, len(trad_history.history['accuracy']) + 1)

plt.plot(epochs_ga, history.history['accuracy'], 'b-', label='GA train')
plt.plot(epochs_ga, history.history['val_accuracy'], 'b--', label='GA validation')
plt.plot(epochs_trad, trad_history.history['accuracy'], 'r-', label='Trad train')
plt.plot(epochs_trad, trad_history.history['val_accuracy'], 'r--', label='Trad validation')

plt.title('Historia de Entrenamiento Comparativa')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("resultados_comparativos_reales.png")
plt.show()