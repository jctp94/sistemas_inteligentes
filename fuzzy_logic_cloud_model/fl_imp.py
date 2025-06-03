import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from numpy.random import normal
from skfuzzy.cluster import cmeans
from datetime import datetime
import os

class CloudModel:
    def __init__(self, Ex, En, He):
        self.Ex = Ex
        self.En = En
        self.He = He

    def membership(self, x, n_drops=100):
        En_prime = normal(self.En, self.He, n_drops)
        mu = np.exp(-(x - self.Ex) ** 2 / (2 * En_prime ** 2))
        return np.mean(mu)

def train_cloud_models(vector, c=3, m=2, error=0.005, maxiter=1000):
    data = vector.values.reshape(1, -1)
    # data:  datos de entrada
    # c:  número de clusters
    # m:  parámetro de la función de membresía
    # error:  error máximo permitido
    # maxiter:  número máximo de iteraciones
    # cntr:  centros de los clusters
    # u:  funciones de membresía
    # _, _, _, _, _, _, _:  variables auxiliares
    cntr, u, _, _, _, _, _ = cmeans(data, c=c, m=m, error=error, maxiter=maxiter)
    models = []
    for i in range(c):
        weights = u[i]
        Ex = cntr[i]
        diffs = (data - Ex).flatten()
        En = np.sqrt(np.average(diffs**2, weights=weights))
        He = np.std(diffs) / 2
        models.append(CloudModel(Ex.item(), En.item(), He.item()))
    return models

# Crear carpeta para guardar gráficos con nombre único por fecha y hora
timestamp_dir = datetime.now().strftime("graficos_%Y%m%d_%H%M%S")
os.makedirs(timestamp_dir, exist_ok=True)

# Cargar datos
df = pd.read_csv("../smart_mobility_dataset.csv")
df = df[["Traffic_Speed_kmh", "Road_Occupancy_%", "Traffic_Condition"]].dropna()

# Variables difusas
speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')
occupancy = ctrl.Antecedent(np.arange(0, 101, 1), 'occupancy')
congestion = ctrl.Consequent(np.arange(0, 101, 1), 'congestion')

# Funciones de membresía actualizadas
speed['low'] = fuzz.trimf(speed.universe, [0, 0, 50])
speed['medium'] = fuzz.trimf(speed.universe, [30, 50, 80])
speed['high'] = fuzz.trimf(speed.universe, [60, 100, 100])

occupancy['low'] = fuzz.trimf(occupancy.universe, [0, 0, 45])
occupancy['medium'] = fuzz.trimf(occupancy.universe, [30, 50, 80])
occupancy['high'] = fuzz.trimf(occupancy.universe, [60, 100, 100])

congestion['low'] = fuzz.trimf(congestion.universe, [0, 0, 40])
congestion['medium'] = fuzz.trimf(congestion.universe, [30, 50, 70])
congestion['high'] = fuzz.trimf(congestion.universe, [60, 100, 100])


# Visualización de funciones de membresía tradicionales
x_vals = np.arange(0, 101, 1)

speed_low = fuzz.trimf(x_vals, [0, 0, 50])
speed_medium = fuzz.trimf(x_vals, [30, 50, 80])
speed_high = fuzz.trimf(x_vals, [60, 100, 100])

occupancy_low = fuzz.trimf(x_vals, [0, 0, 45])
occupancy_medium = fuzz.trimf(x_vals, [30, 50, 80])
occupancy_high = fuzz.trimf(x_vals, [60, 100, 100])

congestion_low = fuzz.trimf(x_vals, [0, 0, 40])
congestion_medium = fuzz.trimf(x_vals, [30, 50, 70])
congestion_high = fuzz.trimf(x_vals, [60, 100, 100])

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x_vals, speed_low, label='Low')
axes[0].plot(x_vals, speed_medium, label='Medium')
axes[0].plot(x_vals, speed_high, label='High')
axes[0].set_title('Fuzzy Membership - Speed')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x_vals, occupancy_low, label='Low')
axes[1].plot(x_vals, occupancy_medium, label='Medium')
axes[1].plot(x_vals, occupancy_high, label='High')
axes[1].set_title('Fuzzy Membership - Occupancy')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(x_vals, congestion_low, label='Low')
axes[2].plot(x_vals, congestion_medium, label='Medium')
axes[2].plot(x_vals, congestion_high, label='High')
axes[2].set_title('Fuzzy Membership - Congestion')
axes[2].legend()
axes[2].grid(True)

axes[2].set_xlabel("Valor")
for ax in axes:
    ax.set_ylabel("Membresía")

plt.tight_layout()
plt.savefig(os.path.join(timestamp_dir, "fuzzy_memberships.png"))

# Reglas difusas actualizadas
rule1 = ctrl.Rule(speed['low'] & occupancy['low'], congestion['medium'])
rule2 = ctrl.Rule(speed['low'] & occupancy['medium'], congestion['high'])
rule3 = ctrl.Rule(speed['low'] & occupancy['high'], congestion['high'])

rule4 = ctrl.Rule(speed['medium'] & occupancy['low'], congestion['low'])
rule5 = ctrl.Rule(speed['medium'] & occupancy['medium'], congestion['medium'])
rule6 = ctrl.Rule(speed['medium'] & occupancy['high'], congestion['high'])

rule7 = ctrl.Rule(speed['high'] & occupancy['low'], congestion['low'])
rule8 = ctrl.Rule(speed['high'] & occupancy['medium'], congestion['medium'])
rule9 = ctrl.Rule(speed['high'] & occupancy['high'], congestion['high'])

# Sistema de inferencia actualizado
congestion_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9
])

cloud_speed_low, cloud_speed_medium, cloud_speed_high = train_cloud_models(df["Traffic_Speed_kmh"])
cloud_occ_low, cloud_occ_medium, cloud_occ_high = train_cloud_models(df["Road_Occupancy_%"])
cong_numeric = df["Traffic_Condition"].map({"Low": 0, "Medium": 50, "High": 100})
cloud_cong_low, cloud_cong_medium, cloud_cong_high = train_cloud_models(cong_numeric) # o fuzzy_output si se prefiere

# Función de predicción con Cloud Model
def cloud_predict(speed_val, occ_val):
    speed_mems = {
        'low': cloud_speed_low.membership(speed_val),
        'medium': cloud_speed_medium.membership(speed_val),
        'high': cloud_speed_high.membership(speed_val),
    }
    occ_mems = {
        'low': cloud_occ_low.membership(occ_val),
        'medium': cloud_occ_medium.membership(occ_val),
        'high': cloud_occ_high.membership(occ_val),
    }

    # Aplicar reglas manualmente
    weights = {
        'low': 0,
        'medium': 50,
        'high': 100,
    }
    rules = [
        ('low', 'low', 'medium'),
        ('low', 'medium', 'high'),
        ('low', 'high', 'high'),
        ('medium', 'low', 'low'),
        ('medium', 'medium', 'medium'),
        ('medium', 'high', 'high'),
        ('high', 'low', 'low'),
        ('high', 'medium', 'medium'),
        ('high', 'high', 'high'),
    ]

    numer = 0
    denom = 0
    for s, o, c in rules:
        activation = speed_mems[s] * occ_mems[o]
        numer += activation * weights[c]
        denom += activation

    if denom == 0:
        return 0
    return int(numer / denom)

# Predicción difusa
def fuzzy_predict(speed_val, occ_val):
    print("fuzzy_predict", speed_val, occ_val)
    congestion_simulator = ctrl.ControlSystemSimulation(congestion_ctrl)
    congestion_simulator.input['speed'] = speed_val
    congestion_simulator.input['occupancy'] = occ_val
    congestion_simulator.compute()
    congestion_value = congestion_simulator.output['congestion']
    print("congestion_value", congestion_value)
    return int(congestion_value)

# Mapeo de clases reales
label_map = {"Low": 0, "Medium": 1, "High": 2}
df["true_label"] = df["Traffic_Condition"].map(label_map)

# Aplicar predicciones
df["fuzzy_output"] = df.apply(lambda row: fuzzy_predict(row["Traffic_Speed_kmh"], row["Road_Occupancy_%"]), axis=1)
df["cloud_output"] = df.apply(lambda row: cloud_predict(row["Traffic_Speed_kmh"], row["Road_Occupancy_%"]), axis=1)

# === SMOTE para balanceo de clases ===
from imblearn.over_sampling import SMOTE

# Aplicar SMOTE para balancear clases
X = df[["Traffic_Speed_kmh", "Road_Occupancy_%"]]
y = df["true_label"]

print("\nAplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Construir nuevo DataFrame con resultados balanceados
df_resampled = pd.DataFrame(X_resampled, columns=["Traffic_Speed_kmh", "Road_Occupancy_%"])
df_resampled["true_label"] = y_resampled

# Recalcular outputs sobre datos balanceados
df_resampled["fuzzy_output"] = df_resampled.apply(lambda row: fuzzy_predict(row["Traffic_Speed_kmh"], row["Road_Occupancy_%"]), axis=1)
df_resampled["cloud_output"] = df_resampled.apply(lambda row: cloud_predict(row["Traffic_Speed_kmh"], row["Road_Occupancy_%"]), axis=1)

# Clasificación de salida continua
def classify_fuzzy_output(val):
    if val <= 33:
        return 0
    elif val <= 66:
        return 1
    else:
        return 2

df_resampled["fuzzy_predicted_label"] = df_resampled["fuzzy_output"].apply(classify_fuzzy_output)
df_resampled["cloud_predicted_label"] = df_resampled["cloud_output"].apply(classify_fuzzy_output)

# Métricas y visualización comparativa
cm_fuzzy = confusion_matrix(df_resampled["true_label"], df_resampled["fuzzy_predicted_label"])
cm_cloud = confusion_matrix(df_resampled["true_label"], df_resampled["cloud_predicted_label"])

report_fuzzy = classification_report(df_resampled["true_label"], df_resampled["fuzzy_predicted_label"], target_names=["Low", "Medium", "High"])
report_cloud = classification_report(df_resampled["true_label"], df_resampled["cloud_predicted_label"], target_names=["Low", "Medium", "High"])

print("=== MÉTRICAS SISTEMA DIFUSO ===")
print("Matriz de confusión:\n", cm_fuzzy)
print("\nReporte de clasificación:\n", report_fuzzy)

print("=== MÉTRICAS CLOUD MODEL ===")
print("Matriz de confusión:\n", cm_cloud)
print("\nReporte de clasificación:\n", report_cloud)

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_fuzzy, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
axes[0].set_title("Confusion Matrix - Fuzzy")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Valor Real")

sns.heatmap(cm_cloud, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
axes[1].set_title("Confusion Matrix - Cloud Model")
axes[1].set_xlabel("Predicción")
axes[1].set_ylabel("Valor Real")

# plt.tight_layout()
# plt.show()

# Guardar matrices de confusión como imágenes
fig_fuzzy, ax_fuzzy = plt.subplots()
sns.heatmap(cm_fuzzy, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax_fuzzy)
ax_fuzzy.set_title("Confusion Matrix - Fuzzy")
ax_fuzzy.set_xlabel("Predicción")
ax_fuzzy.set_ylabel("Valor Real")
fig_fuzzy.tight_layout()
fig_fuzzy.savefig(os.path.join(timestamp_dir, "confusion_matrix_fuzzy.png"))

fig_cloud, ax_cloud = plt.subplots()
sns.heatmap(cm_cloud, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax_cloud)
ax_cloud.set_title("Confusion Matrix - Cloud Model")
ax_cloud.set_xlabel("Predicción")
ax_cloud.set_ylabel("Valor Real")
fig_cloud.tight_layout()
fig_cloud.savefig(os.path.join(timestamp_dir, "confusion_matrix_cloud.png"))

# Exportar resultados a un archivo .txt
timestamp = datetime.now().strftime("%H%M%S")
with open(os.path.join(timestamp_dir, f"classification_reports_{timestamp}.txt"), "w") as f:
    f.write("=== MÉTRICAS SISTEMA DIFUSO ===\n")
    f.write("Matriz de confusión:\n")
    f.write(np.array2string(cm_fuzzy))
    f.write("\n\nReporte de clasificación:\n")
    f.write(report_fuzzy)
    f.write("\n\n=== MÉTRICAS CLOUD MODEL ===\n")
    f.write("Matriz de confusión:\n")
    f.write(np.array2string(cm_cloud))
    f.write("\n\nReporte de clasificación:\n")
    f.write(report_cloud)

# Visualización de funciones de membresía del Cloud Model
x_vals = np.linspace(0, 100, 500)
def cloud_mf(Ex, En, He, x_vals, n_drops=100):
    En_prime = np.random.normal(En, He, n_drops)
    return np.array([np.mean(np.exp(-(x - Ex) ** 2 / (2 * En_prime ** 2))) for x in x_vals])

cloud_speed_low_vals = cloud_mf(0, 15, 3, x_vals)
cloud_speed_med_vals = cloud_mf(50, 15, 3, x_vals)
cloud_speed_high_vals = cloud_mf(100, 15, 3, x_vals)

cloud_occ_low_vals = cloud_mf(0, 15, 3, x_vals)
cloud_occ_med_vals = cloud_mf(50, 15, 3, x_vals)
cloud_occ_high_vals = cloud_mf(100, 15, 3, x_vals)

cloud_cong_low_vals = cloud_mf(0, 15, 3, x_vals)
cloud_cong_med_vals = cloud_mf(50, 15, 3, x_vals)
cloud_cong_high_vals = cloud_mf(100, 15, 3, x_vals)


# Graficar
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x_vals, cloud_speed_low_vals, label='Speed Low')
axes[0].plot(x_vals, cloud_speed_med_vals, label='Speed Medium')
axes[0].plot(x_vals, cloud_speed_high_vals, label='Speed High')
axes[0].set_title('Cloud Model - Speed')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x_vals, cloud_occ_low_vals, label='Occupancy Low')
axes[1].plot(x_vals, cloud_occ_med_vals, label='Occupancy Medium')
axes[1].plot(x_vals, cloud_occ_high_vals, label='Occupancy High')
axes[1].set_title('Cloud Model - Occupancy')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(x_vals, cloud_cong_low_vals, label='Congestion Low')
axes[2].plot(x_vals, cloud_cong_med_vals, label='Congestion Medium')
axes[2].plot(x_vals, cloud_cong_high_vals, label='Congestion High')
axes[2].set_title('Cloud Model - Congestion')
axes[2].legend()
axes[2].grid(True)

axes[2].set_xlabel("Valor")
for ax in axes:
    ax.set_ylabel("Membresía")

plt.tight_layout()
plt.savefig(os.path.join(timestamp_dir, "cloud_memberships.png"))


# === Experimentos variando parámetros del modelo Cloud ===
import itertools
import csv

def run_experiments():
    n_drops_list = [50, 100, 150]
    En_list = [10, 15, 20]
    He_list = [2, 5, 10]

    experiment_results = []
    combs = list(itertools.product(n_drops_list, En_list, He_list))

    for n_drops, En, He in combs:
        print(f"\nEjecutando experimento con n_drops={n_drops}, En={En}, He={He}")
        def cloud_mf_param(Ex, x_vals):
            En_prime = np.random.normal(En, He, n_drops)
            return np.array([np.mean(np.exp(-(x - Ex)**2 / (2 * En_prime**2))) for x in x_vals])

        # Redefinir modelos de nube con estos parámetros
        global cloud_speed_low, cloud_speed_medium, cloud_speed_high
        global cloud_occ_low, cloud_occ_medium, cloud_occ_high

        cloud_speed_low = CloudModel(0, En, He)
        cloud_speed_medium = CloudModel(50, En, He)
        cloud_speed_high = CloudModel(100, En, He)

        cloud_occ_low = CloudModel(0, En, He)
        cloud_occ_medium = CloudModel(50, En, He)
        cloud_occ_high = CloudModel(100, En, He)

        df_temp = df_resampled.copy()
        df_temp["cloud_output"] = df_temp.apply(lambda row: cloud_predict(row["Traffic_Speed_kmh"], row["Road_Occupancy_%"]), axis=1)
        df_temp["cloud_predicted_label"] = df_temp["cloud_output"].apply(classify_fuzzy_output)

        cm_cloud_exp = confusion_matrix(df_temp["true_label"], df_temp["cloud_predicted_label"])
        report_cloud_exp = classification_report(df_temp["true_label"], df_temp["cloud_predicted_label"], output_dict=True)

        experiment_results.append({
            "n_drops": n_drops,
            "En": En,
            "He": He,
            "accuracy": report_cloud_exp["accuracy"],
            "f1_macro": report_cloud_exp["macro avg"]["f1-score"],
            "precision_macro": report_cloud_exp["macro avg"]["precision"],
            "recall_macro": report_cloud_exp["macro avg"]["recall"]
        })

    csv_path = os.path.join(timestamp_dir, "experimentos_cloud_model.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=experiment_results[0].keys())
        writer.writeheader()
        writer.writerows(experiment_results)

    print(f"\nResultados guardados en {csv_path}")

if __name__ == "__main__":
    run_experiments()