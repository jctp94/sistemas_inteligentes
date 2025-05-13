import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Cargar y preparar datos
df = pd.read_csv("smart_mobility_dataset.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
df['Is_Rush_Hour'] = (((df['Hour'] >= 7) & (df['Hour'] <= 9)) |
                      ((df['Hour'] >= 16) & (df['Hour'] <= 19))).astype(int)

# Variables relevantes
df_clean = df[['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%',
               'Weather_Condition', 'Hour', 'Day_of_Week',
               'Is_Weekend', 'Is_Rush_Hour', 'Traffic_Condition']]

# Discretización
def discretize_column(df, column, n_bins=3):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df.loc[:, f"{column}_D"] = discretizer.fit_transform(df[[column]]).astype(int)
    bin_edges = discretizer.bin_edges_[0]
    return {i: f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)}

bin_mappings = {}
for col in ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 'Hour']:
    bin_mappings[col] = discretize_column(df_clean, col)

# Mapear variables categóricas
weather_map = {val: i for i, val in enumerate(df_clean['Weather_Condition'].unique())}
reverse_weather = {i: val for val, i in weather_map.items()}
df_clean.loc[:, 'Weather_Condition_D'] = df_clean['Weather_Condition'].map(weather_map)

traffic_map = {val: i for i, val in enumerate(df_clean['Traffic_Condition'].unique())}
reverse_traffic = {i: val for val, i in traffic_map.items()}
df_clean.loc[:, 'Traffic_Condition_D'] = df_clean['Traffic_Condition'].map(traffic_map)

# Log-Loss Evaluation
features = ['Hour_D', 'Is_Rush_Hour', 'Weather_Condition_D']
X = df_clean[features]
y = df_clean['Traffic_Condition_D']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

prob_table = X_train.copy()
prob_table['Traffic_Condition_D'] = y_train

# Inferencia
def get_conditional_probabilities(x_row):
    subset = prob_table[
        (prob_table['Hour_D'] == x_row['Hour_D']) &
        (prob_table['Is_Rush_Hour'] == x_row['Is_Rush_Hour']) &
        (prob_table['Weather_Condition_D'] == x_row['Weather_Condition_D'])
    ]
    probs = subset['Traffic_Condition_D'].value_counts(normalize=True)
    full_probs = np.zeros(len(reverse_traffic))
    for i in range(len(reverse_traffic)):
        full_probs[i] = probs.get(i, 1e-6)
    return full_probs

y_pred_proba = np.vstack(X_test.apply(get_conditional_probabilities, axis=1))
y_true = y_test.values
log_losses = [log_loss([yt], [yp], labels=list(range(len(reverse_traffic)))) for yt, yp in zip(y_true, y_pred_proba)]

# Graficar Log-Loss
plt.figure(figsize=(10, 6))
plt.plot(log_losses, label='Log-Loss por muestra')
plt.axhline(np.mean(log_losses), color='red', linestyle='--', label=f'Promedio: {np.mean(log_losses):.4f}')
plt.title('Log-Loss por muestra en red bayesiana')
plt.xlabel('Índice de muestra')
plt.ylabel('Log-Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('log_loss_plot.png')
plt.show()
