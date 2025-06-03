import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.patches import Patch

# 1. Cargar y preparar datos
print("Cargando y procesando datos...")
df = pd.read_csv('smart_mobility_dataset.csv')

# Procesar timestamp y crear variables temporales
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
df['Is_Rush_Hour'] = (((df['Hour'] >= 7) & (df['Hour'] <= 9)) | 
                     ((df['Hour'] >= 16) & (df['Hour'] <= 19))).astype(int)

# Seleccionar variables relevantes
df_clean = df[['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 
              'Weather_Condition', 'Hour', 'Day_of_Week', 
              'Is_Weekend', 'Is_Rush_Hour', 'Traffic_Condition']]

# 2. Discretizar variables continuas
print("Discretizando variables...")
def discretize_column(df, column, n_bins=3):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df[f"{column}_D"] = discretizer.fit_transform(df[[column]]).astype(int)
    
    # Crear mapeo para interpretabilidad
    bin_edges = discretizer.bin_edges_[0]
    bin_mapping = {}
    for i in range(len(bin_edges)-1):
        bin_mapping[i] = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
    
    return bin_mapping

# Discretizar variables numéricas
bin_mappings = {}
for col in ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 'Hour']:
    bin_mappings[col] = discretize_column(df_clean, col)

# 3. Codificar variables categóricas
# Para Weather_Condition
if df_clean['Weather_Condition'].dtype == 'object':
    weather_mapping = {val: i for i, val in enumerate(df_clean['Weather_Condition'].unique())}
    df_clean['Weather_Condition_D'] = df_clean['Weather_Condition'].map(weather_mapping)
    reverse_weather = {i: val for val, i in weather_mapping.items()}
else:
    df_clean['Weather_Condition_D'] = df_clean['Weather_Condition']
    reverse_weather = {i: f"Tipo {i}" for i in df_clean['Weather_Condition_D'].unique()}

# Para Traffic_Condition
if df_clean['Traffic_Condition'].dtype == 'object':
    traffic_mapping = {val: i for i, val in enumerate(df_clean['Traffic_Condition'].unique())}
    df_clean['Traffic_Condition_D'] = df_clean['Traffic_Condition'].map(traffic_mapping)
    reverse_traffic = {i: val for val, i in traffic_mapping.items()}
else:
    df_clean['Traffic_Condition_D'] = df_clean['Traffic_Condition']
    reverse_traffic = {i: f"Nivel {i}" for i in df_clean['Traffic_Condition_D'].unique()}

# 4. Definir estructura de la red bayesiana manualmente
G = nx.DiGraph()

# Nodos principales (usar nombres cortos para la visualización)
G.add_nodes_from(['Traffic', 'Speed', 'Volume', 'Occupancy', 'Weather', 'Hour', 'Rush_Hour'])

# Definir relaciones causales
edges = [
    # Factores que afectan la congestión
    ('Weather', 'Traffic'),
    ('Hour', 'Traffic'),
    ('Rush_Hour', 'Traffic'),
    
    # Efectos de la congestión
    ('Traffic', 'Speed'),
    ('Traffic', 'Volume'),
    ('Traffic', 'Occupancy'),
    
    # Relaciones entre variables temporales
    ('Hour', 'Rush_Hour')
]
G.add_edges_from(edges)

# 5. Visualizar la red bayesiana
print("Visualizando la red bayesiana...")
plt.figure(figsize=(12, 10))

# Crear layout más legible
pos = {
    'Traffic': (0.5, 0.5),    # Centro
    'Speed': (0.8, 0.3),      # Abajo derecha
    'Volume': (0.2, 0.3),     # Abajo izquierda
    'Occupancy': (0.5, 0.1),  # Abajo
    'Weather': (0.2, 0.7),    # Arriba izquierda
    'Hour': (0.5, 0.9),       # Arriba
    'Rush_Hour': (0.8, 0.7)   # Arriba derecha
}

# Colorear nodos por categoría
node_colors = {
    'Traffic': 'lightcoral',      # Variable objetivo
    'Speed': 'lightblue',         # Variables de tráfico
    'Volume': 'lightblue',
    'Occupancy': 'lightblue',
    'Weather': 'lightyellow',     # Variables externas 
    'Hour': 'lightgreen',         # Variables temporales
    'Rush_Hour': 'lightgreen'
}

# Dibujar nodos con colores
nx.draw_networkx_nodes(G, pos, 
                     node_color=[node_colors[node] for node in G.nodes()],
                     node_size=3000, alpha=0.9, linewidths=2, edgecolors='gray')

# Dibujar etiquetas de nodos
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# Dibujar bordes con flechas
nx.draw_networkx_edges(G, pos, 
                     arrowstyle='->', arrowsize=25, 
                     edge_color='darkgray', width=2.0, 
                     connectionstyle='arc3,rad=0.1')

# Crear leyenda
legend_elements = [
    Patch(facecolor='lightcoral', edgecolor='gray', label='Variable objetivo'),
    Patch(facecolor='lightblue', edgecolor='gray', label='Variables de tráfico'),
    Patch(facecolor='lightyellow', edgecolor='gray', label='Variables externas'),
    Patch(facecolor='lightgreen', edgecolor='gray', label='Variables temporales')
]

plt.legend(handles=legend_elements, loc='lower left', fontsize=12)
plt.title('Red Bayesiana para Predicción de Congestión de Tráfico', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('red_bayesiana_trafico.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Calcular tabla de probabilidad condicional para Traffic_Condition
print("\nCalculando tabla de probabilidad condicional para Traffic_Condition...")

# Probabilidad a priori de Traffic_Condition
prior_prob = df_clean['Traffic_Condition_D'].value_counts(normalize=True)
print("Probabilidades a priori (sin evidencia):")
for traffic_class, prob in prior_prob.items():
    traffic_name = reverse_traffic.get(traffic_class, f"Nivel {traffic_class}")
    print(f"  {traffic_name}: {prob:.4f}")

# 7. Realizar inferencia
print("\nRealizando inferencia con evidencia...")

# Definir la evidencia
hour_value = 2  # Valor para Hour_D (intervalo específico)
rush_hour_value = 1  # Es hora punta
weather_value = 1  # Valor para Weather_Condition_D

# Describir la evidencia de manera interpretable
evidence_desc = {
    'Hour': f"{hour_value} ({bin_mappings['Hour'].get(hour_value, 'Desconocido')})",
    'Is_Rush_Hour': rush_hour_value,
    'Weather_Condition': f"{weather_value} ({reverse_weather.get(weather_value, 'Desconocido')})"
}

print("Evidencia utilizada:")
for var, val in evidence_desc.items():
    print(f"  {var} = {val}")

# Filtrar el dataset con la evidencia
filtered_data = df_clean[
    (df_clean['Hour_D'] == hour_value) & 
    (df_clean['Is_Rush_Hour'] == rush_hour_value) & 
    (df_clean['Weather_Condition_D'] == weather_value)
]

# Si no hay suficientes datos para la combinación exacta, relajar las condiciones
if len(filtered_data) < 10:
    print("Pocos datos para la combinación exacta, relajando condiciones...")
    filtered_data = df_clean[
        (df_clean['Hour_D'] == hour_value) & 
        (df_clean['Is_Rush_Hour'] == rush_hour_value)
    ]

# Calcular probabilidades condicionales
if len(filtered_data) > 0:
    conditional_probs = filtered_data['Traffic_Condition_D'].value_counts(normalize=True)
    
    # Asegurar que todos los niveles de tráfico estén representados
    for i in range(len(reverse_traffic)):
        if i not in conditional_probs:
            conditional_probs[i] = 0.0
            
    # Ordenar por índice
    conditional_probs = conditional_probs.sort_index()
    
    # 8. Visualizar resultados
    plt.figure(figsize=(10, 6))
    
    traffic_labels = [reverse_traffic.get(i, f"Nivel {i}") for i in conditional_probs.index]
    
    # Dibujar barras con colores gradientes
    colors = plt.cm.viridis(conditional_probs.values)
    bars = plt.bar(traffic_labels, conditional_probs.values, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Añadir etiquetas con valores
    for bar, prob in zip(bars, conditional_probs.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{prob:.3f}", ha='center', va='bottom', fontweight='bold')
    
    # Formatear gráfico
    evidence_str = ', '.join([f"{k}={v}" for k, v in evidence_desc.items()])
    plt.title(f'Probabilidad de Congestión de Tráfico\ndada la evidencia: {evidence_str}', fontsize=14)
    plt.xlabel('Nivel de Congestión', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('inferencia_trafico.png', dpi=300)
    plt.show()
    
    # Imprimir resultados
    print("\nProbabilidades condicionales dado el escenario:")
    for idx, prob in conditional_probs.items():
        traffic_name = reverse_traffic.get(idx, f"Nivel {idx}")
        print(f"  {traffic_name}: {prob:.4f}")
else:
    print("No hay suficientes datos para realizar la inferencia con esta evidencia.")

print("\nAnálisis completado. Imágenes guardadas en 'red_bayesiana_trafico.png' y 'inferencia_trafico.png'")