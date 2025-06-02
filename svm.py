import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score, 
                                   StratifiedKFold, cross_validate, cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, precision_recall_curve, roc_curve, auc, 
                           precision_score, recall_score, make_scorer)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import time
import joblib
from itertools import cycle
import gc

warnings.filterwarnings('ignore')

# Configurar estilo de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================================
# 1. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO CON K-FOLD
# ==================================
def train_svm_with_kfold(n_splits=5):
    """
    Entrena SVM con K-Fold Cross Validation
    """
    print("=" * 70)
    print("SVM MULTICLASE CON K-FOLD CROSS VALIDATION")
    print("=" * 70)
    
    # 1. Cargar y preprocesar datos
    print("\n1. Cargando y preprocesando dataset...")
    df = pd.read_csv('smart_mobility_dataset.csv')
    print(f"   - Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Procesar timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    df['Is_Rush_Hour'] = (((df['Hour'] >= 7) & (df['Hour'] <= 9)) | 
                          ((df['Hour'] >= 16) & (df['Hour'] <= 19))).astype(int)
    
    # Crear características adicionales
    df['Speed_Occupancy_Ratio'] = df['Traffic_Speed_kmh'] / (df['Road_Occupancy_%'] + 1)
    df['Congestion_Index'] = (df['Vehicle_Count'] * df['Road_Occupancy_%']) / 100
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Eliminar columnas irrelevantes
    df_clean = df.drop(['Timestamp', 'Emission_Levels_g_km', 'Energy_Consumption_L_h'], axis=1)
    
    # Codificar variables categóricas
    print("\n2. Codificando variables categóricas...")
    label_encoders = {}
    
    # Weather Condition
    le_weather = LabelEncoder()
    df_clean['Weather_Condition_Encoded'] = le_weather.fit_transform(df_clean['Weather_Condition'])
    label_encoders['Weather_Condition'] = le_weather
    
    # Traffic Light State
    le_traffic_light = LabelEncoder()
    df_clean['Traffic_Light_State_Encoded'] = le_traffic_light.fit_transform(df_clean['Traffic_Light_State'])
    label_encoders['Traffic_Light_State'] = le_traffic_light
    
    # Variable objetivo
    le_traffic_condition = LabelEncoder()
    df_clean['Traffic_Condition_Encoded'] = le_traffic_condition.fit_transform(df_clean['Traffic_Condition'])
    label_encoders['Traffic_Condition'] = le_traffic_condition
    
    print(f"   - Clases de congestión: {list(le_traffic_condition.classes_)}")
    
    # Preparar características
    feature_columns = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 
                      'Weather_Condition_Encoded', 'Traffic_Light_State_Encoded',
                      'Hour', 'Day_of_Week', 'Is_Weekend', 'Is_Rush_Hour',
                      'Latitude', 'Longitude', 'Accident_Report', 'Sentiment_Score',
                      'Ride_Sharing_Demand', 'Parking_Availability',
                      'Speed_Occupancy_Ratio', 'Congestion_Index', 
                      'Hour_sin', 'Hour_cos']
    
    X = df_clean[feature_columns]
    y = df_clean['Traffic_Condition_Encoded']
    
    # División inicial para tener un conjunto de test fijo
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n3. División de datos:")
    print(f"   - Train+Val: {X_train_val.shape}")
    print(f"   - Test (holdout): {X_test.shape}")
    
    # ==================================
    # 2. CONFIGURAR K-FOLD CROSS VALIDATION
    # ==================================
    print(f"\n4. Configurando {n_splits}-Fold Cross Validation...")
    
    # Crear pipeline con escalado y SMOTE
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])
    
    # Parámetros para GridSearchCV
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    # Configurar K-Fold estratificado
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # GridSearchCV con K-Fold
    print("\n5. Iniciando búsqueda de hiperparámetros con K-Fold...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Entrenar
    start_time = time.time()
    grid_search.fit(X_train_val, y_train_val)
    training_time = time.time() - start_time
    
    print(f"\n   Búsqueda completada en {training_time:.2f} segundos")
    print(f"   - Mejores parámetros: {grid_search.best_params_}")
    print(f"   - Mejor score F1 (macro) en CV: {grid_search.best_score_:.4f}")
    
    # ==================================
    # 3. EVALUACIÓN DETALLADA CON K-FOLD
    # ==================================
    print(f"\n6. Realizando evaluación detallada con {n_splits}-Fold CV...")
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Métricas múltiples con cross_validate
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    }
    
    cv_results = cross_validate(
        best_model, X_train_val, y_train_val, 
        cv=skf, scoring=scoring, return_train_score=True
    )
    
    # Mostrar resultados de K-Fold
    print("\n   Resultados de K-Fold Cross Validation:")
    print("   " + "-" * 50)
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        val_scores = cv_results[f'test_{metric}']
        print(f"   {metric}:")
        print(f"     - Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
        print(f"     - Val:   {val_scores.mean():.4f} (+/- {val_scores.std() * 2:.4f})")
    
    # Obtener predicciones de cross-validation para análisis
    print("\n7. Generando predicciones cross-validated...")
    y_pred_cv = cross_val_predict(best_model, X_train_val, y_train_val, cv=skf)
    
    # ==================================
    # 4. EVALUACIÓN EN CONJUNTO DE TEST
    # ==================================
    print("\n8. Evaluando en conjunto de test (holdout)...")
    
    # Reentrenar en todo el conjunto train+val
    best_model.fit(X_train_val, y_train_val)
    
    # Predicciones en test
    y_pred_test = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Métricas en test
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n   Métricas en conjunto de test:")
    print(f"   - Accuracy: {test_accuracy:.4f}")
    print(f"   - F1-Score (macro): {test_f1_macro:.4f}")
    print(f"   - F1-Score (weighted): {test_f1_weighted:.4f}")
    
    # Reporte detallado
    print("\n   Reporte de clasificación (Test Set):")
    class_names = le_traffic_condition.classes_
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Guardar resultados
    results = {
        'best_model': best_model,
        'grid_search': grid_search,
        'cv_results': cv_results,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'X_train_val': X_train_val,
        'y_train_val': y_train_val,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'y_pred_proba': y_pred_proba,
        'y_pred_cv': y_pred_cv,
        'class_names': class_names,
        'n_splits': n_splits
    }
    
    return results

# ==================================
# 5. FUNCIONES DE VISUALIZACIÓN
# ==================================
def plot_kfold_results(results):
    """Visualiza los resultados de K-Fold Cross Validation"""
    print("\n9. Generando visualizaciones de K-Fold...")
    
    cv_results = results['cv_results']
    n_splits = results['n_splits']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Boxplot de métricas por fold
    ax1 = axes[0, 0]
    metrics_data = []
    metrics_names = []
    
    for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        test_scores = cv_results[f'test_{metric}']
        metrics_data.extend(test_scores)
        metrics_names.extend([metric] * len(test_scores))
    
    df_metrics = pd.DataFrame({
        'Score': metrics_data,
        'Metric': metrics_names
    })
    
    sns.boxplot(data=df_metrics, x='Metric', y='Score', ax=ax1)
    ax1.set_title(f'Distribución de Métricas en {n_splits}-Fold CV', fontsize=14)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparación Train vs Validation
    ax2 = axes[0, 1]
    metrics = ['accuracy', 'f1_macro']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_means = [cv_results[f'train_{m}'].mean() for m in metrics]
    val_means = [cv_results[f'test_{m}'].mean() for m in metrics]
    train_stds = [cv_results[f'train_{m}'].std() for m in metrics]
    val_stds = [cv_results[f'test_{m}'].std() for m in metrics]
    
    bars1 = ax2.bar(x - width/2, train_means, width, yerr=train_stds, 
                     label='Train', capsize=5, color='skyblue')
    bars2 = ax2.bar(x + width/2, val_means, width, yerr=val_stds, 
                     label='Validation', capsize=5, color='lightcoral')
    
    ax2.set_xlabel('Métricas')
    ax2.set_ylabel('Score')
    ax2.set_title('Comparación Train vs Validation', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0.7, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Scores por Fold
    ax3 = axes[1, 0]
    folds = range(1, n_splits + 1)
    
    for metric in ['accuracy', 'f1_macro']:
        scores = cv_results[f'test_{metric}']
        ax3.plot(folds, scores, 'o-', label=metric, markersize=8)
    
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Score')
    ax3.set_title(f'Scores por Fold ({n_splits}-Fold CV)', fontsize=14)
    ax3.set_xticks(folds)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.7, 1.0)
    
    # 4. Matriz de confusión de CV predictions
    ax4 = axes[1, 1]
    cm_cv = confusion_matrix(results['y_train_val'], results['y_pred_cv'])
    sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', 
                xticklabels=results['class_names'], 
                yticklabels=results['class_names'], ax=ax4)
    ax4.set_title('Matriz de Confusión (Cross-Validation)', fontsize=14)
    ax4.set_ylabel('Clase Real')
    ax4.set_xlabel('Clase Predicha')
    
    plt.tight_layout()
    plt.savefig('svm_kfold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Análisis K-Fold guardado en 'svm_kfold_analysis.png'")

def plot_confusion_matrix_detailed(results):
    """Genera matriz de confusión detallada del conjunto de test"""
    print("\nGenerando matriz de confusión detallada...")
    
    plt.figure(figsize=(10, 8))
    
    y_test = results['y_test']
    y_pred = results['y_pred_test']
    class_names = results['class_names']
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear anotaciones con conteos y porcentajes
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cantidad'})
    
    plt.title('Matriz de Confusión - Conjunto de Test', fontsize=16)
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    
    # Añadir métricas por clase
    for i, class_name in enumerate(class_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.text(len(class_names) + 0.5, i + 0.5, 
                f'P: {precision:.2f}\nR: {recall:.2f}\nF1: {f1:.2f}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Matriz de confusión guardada en 'svm_confusion_matrix_detailed.png'")

def plot_roc_curves_multiclass(results):
    """Genera curvas ROC para cada clase"""
    print("\nGenerando curvas ROC multiclase...")
    
    plt.figure(figsize=(10, 8))
    
    y_test = results['y_test']
    y_pred_proba = results['y_pred_proba']
    class_names = results['class_names']
    
    # Binarizar las etiquetas
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Calcular ROC curve y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC Multiclase', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Curvas ROC guardadas en 'svm_roc_curves.png'")

def plot_feature_analysis(results):
    """Análisis de importancia de características"""
    print("\nGenerando análisis de características...")
    
    # Obtener el modelo SVM del pipeline
    pipeline = results['best_model']
    scaler = pipeline.named_steps['scaler']
    svm_model = pipeline.named_steps['svm']
    
    # Para aproximar importancia, entrenar un SVM lineal
    X_train_scaled = scaler.transform(results['X_train_val'])
    linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
    linear_svm.fit(X_train_scaled, results['y_train_val'])
    
    # Obtener coeficientes (importancia aproximada)
    feature_importance = np.abs(linear_svm.coef_[0])
    feature_names = results['feature_columns']
    
    # Crear DataFrame para mejor manejo
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Visualización
    plt.figure(figsize=(12, 8))
    
    top_n = 15
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = plt.barh(range(top_n), top_features['importance'], color=colors)
    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('Importancia Relativa', fontsize=12)
    plt.title(f'Top {top_n} Características más Importantes (Aproximación con SVM Lineal)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Añadir valores
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'], i, f' {row["importance"]:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('svm_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Importancia de características guardada en 'svm_feature_importance.png'")

def plot_probability_calibration(results):
    """Análisis de calibración de probabilidades"""
    print("\nGenerando análisis de calibración de probabilidades...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    y_test = results['y_test']
    y_pred_proba = results['y_pred_proba']
    class_names = results['class_names']
    
    # 1. Histograma de probabilidades máximas
    ax1 = axes[0]
    max_probs = np.max(y_pred_proba, axis=1)
    y_pred = results['y_pred_test']
    correct = y_pred == y_test
    
    ax1.hist(max_probs[correct], bins=20, alpha=0.5, label='Correctas', 
             density=True, color='green', edgecolor='black')
    ax1.hist(max_probs[~correct], bins=20, alpha=0.5, label='Incorrectas', 
             density=True, color='red', edgecolor='black')
    
    ax1.axvline(max_probs[correct].mean(), color='green', linestyle='--', 
                label=f'Media Correctas: {max_probs[correct].mean():.3f}')
    ax1.axvline(max_probs[~correct].mean(), color='red', linestyle='--', 
                label=f'Media Incorrectas: {max_probs[~correct].mean():.3f}')
    
    ax1.set_xlabel('Probabilidad Máxima', fontsize=12)
    ax1.set_ylabel('Densidad', fontsize=12)
    ax1.set_title('Distribución de Confianza del Modelo', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Calibración por bins
    ax2 = axes[1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Para cada clase
    for i, class_name in enumerate(class_names):
        # Obtener probabilidades para esta clase
        class_probs = y_pred_proba[:, i]
        class_true = (y_test == i).astype(int)
        
        # Calcular calibración
        bin_accuracies = []
        bin_counts = []
        
        for j in range(n_bins):
            mask = (class_probs >= bin_edges[j]) & (class_probs < bin_edges[j+1])
            if mask.sum() > 0:
                bin_acc = class_true[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)
        
        # Plot calibración
        valid_bins = ~np.isnan(bin_accuracies)
        ax2.plot(bin_centers[valid_bins], np.array(bin_accuracies)[valid_bins], 
                'o-', label=class_name, markersize=8)
    
    # Línea perfecta de calibración
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
    ax2.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax2.set_ylabel('Fracción de Positivos', fontsize=12)
    ax2.set_title('Curva de Calibración por Clase', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('svm_probability_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Análisis de calibración guardado en 'svm_probability_calibration.png'")

def plot_comprehensive_report(results):
    """Genera un reporte visual completo"""
    print("\nGenerando reporte visual completo...")
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    # 1. Matriz de confusión
    ax1 = fig.add_subplot(gs[0, :2])
    cm = confusion_matrix(results['y_test'], results['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=results['class_names'], 
                yticklabels=results['class_names'], ax=ax1)
    ax1.set_title('Matriz de Confusión - Test Set', fontsize=14)
    ax1.set_ylabel('Real')
    ax1.set_xlabel('Predicho')
    
    # 2. Métricas por clase
    ax2 = fig.add_subplot(gs[0, 2])
    report = classification_report(results['y_test'], results['y_pred_test'], 
                                 target_names=results['class_names'], output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    class_names = results['class_names']
    
    data = []
    for metric in metrics:
        for class_name in class_names:
            data.append([metric, class_name, report[class_name][metric]])
    
    df_metrics = pd.DataFrame(data, columns=['Metric', 'Class', 'Score'])
    pivot_df = df_metrics.pivot(index='Class', columns='Metric', values='Score')
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Métricas por Clase', fontsize=14)
    
    # 3. K-Fold scores
    ax3 = fig.add_subplot(gs[1, :])
    cv_results = results['cv_results']
    metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    positions = np.arange(len(metrics_to_plot))
    for i, metric in enumerate(metrics_to_plot):
        scores = cv_results[f'test_{metric}']
        parts = ax3.violinplot([scores], positions=[i], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(plt.cm.Set3(i))
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(metrics_to_plot)
    ax3.set_ylabel('Score')
    ax3.set_title(f'{results["n_splits"]}-Fold Cross Validation Results', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. ROC Curves
    ax4 = fig.add_subplot(gs[2, :2])
    y_test_bin = label_binarize(results['y_test'], classes=[0, 1, 2])
    
    for i, class_name in enumerate(results['class_names']):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], results['y_pred_proba'][:, i])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', lw=2)
    ax4.set_xlabel('FPR')
    ax4.set_ylabel('TPR')
    ax4.set_title('Curvas ROC', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribución de predicciones
    ax5 = fig.add_subplot(gs[2, 2])
    pred_counts = pd.Series(results['y_pred_test']).value_counts().sort_index()
    true_counts = pd.Series(results['y_test']).value_counts().sort_index()
    
    x = np.arange(len(results['class_names']))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, true_counts.values, width, label='Real', alpha=0.8)
    bars2 = ax5.bar(x + width/2, pred_counts.values, width, label='Predicho', alpha=0.8)
    
    ax5.set_xlabel('Clase')
    ax5.set_ylabel('Cantidad')
    ax5.set_title('Distribución Real vs Predicha', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(results['class_names'])
    ax5.legend()
    
    # 6. Análisis temporal (si es posible)
    ax6 = fig.add_subplot(gs[3, :])
    if 'Hour' in results['X_test'].columns:
        error_analysis = pd.DataFrame({
            'Hour': results['X_test']['Hour'],
            'Error': results['y_pred_test'] != results['y_test']
        })
        
        error_by_hour = error_analysis.groupby('Hour')['Error'].agg(['sum', 'count'])
        error_by_hour['error_rate'] = error_by_hour['sum'] / error_by_hour['count']
        
        ax6.plot(error_by_hour.index, error_by_hour['error_rate'], 'o-', linewidth=2, markersize=8)
        ax6.set_xlabel('Hora del Día')
        ax6.set_ylabel('Tasa de Error')
        ax6.set_title('Tasa de Error por Hora del Día', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(range(0, 24, 2))
    
    # 7. Resumen de métricas
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis('off')
    
    summary_text = f"""
    RESUMEN DE RESULTADOS - SVM con {results['n_splits']}-Fold Cross Validation
    
    Conjunto de Test (Holdout):
    • Accuracy: {accuracy_score(results['y_test'], results['y_pred_test']):.4f}
    • F1-Score (macro): {f1_score(results['y_test'], results['y_pred_test'], average='macro'):.4f}
    • F1-Score (weighted): {f1_score(results['y_test'], results['y_pred_test'], average='weighted'):.4f}
    
    Cross-Validation ({results['n_splits']} folds):
    • Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std()*2:.4f}
    • F1-Score (macro): {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std()*2:.4f}
    
    Mejores Hiperparámetros:
    • {results['grid_search'].best_params_}
    
    Tiempo de entrenamiento: {results['grid_search'].refit_time_:.2f} segundos
    """
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Análisis Completo - SVM para Predicción de Congestión de Tráfico', 
                 fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.savefig('svm_comprehensive_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Reporte completo guardado en 'svm_comprehensive_report.png'")

# ==================================
# 6. FUNCIÓN PRINCIPAL
# ==================================
def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    try:
        # Entrenar modelo con K-Fold
        results = train_svm_with_kfold(n_splits=5)  # Puedes cambiar n_splits aquí
        
        print("\n10. Generando todas las visualizaciones...")
        
        # Generar visualizaciones
        plot_kfold_results(results)
        gc.collect()
        
        plot_confusion_matrix_detailed(results)
        gc.collect()
        
        plot_roc_curves_multiclass(results)
        gc.collect()
        
        plot_feature_analysis(results)
        gc.collect()
        
        plot_probability_calibration(results)
        gc.collect()
        
        plot_comprehensive_report(results)
        gc.collect()
        
        # Guardar modelo
        print("\n11. Guardando modelo y resultados...")
        joblib.dump(results['best_model'], 'svm_best_model_kfold.pkl')
        joblib.dump(results['label_encoders'], 'label_encoders.pkl')
        joblib.dump(results['feature_columns'], 'feature_columns.pkl')
        
        # Guardar métricas en JSON
        import json
        metrics_summary = {
            'test_accuracy': float(accuracy_score(results['y_test'], results['y_pred_test'])),
            'test_f1_macro': float(f1_score(results['y_test'], results['y_pred_test'], average='macro')),
            'test_f1_weighted': float(f1_score(results['y_test'], results['y_pred_test'], average='weighted')),
            'cv_accuracy_mean': float(results['cv_results']['test_accuracy'].mean()),
            'cv_accuracy_std': float(results['cv_results']['test_accuracy'].std()),
            'cv_f1_macro_mean': float(results['cv_results']['test_f1_macro'].mean()),
            'cv_f1_macro_std': float(results['cv_results']['test_f1_macro'].std()),
            'best_params': results['grid_search'].best_params_,
            'n_splits': results['n_splits']
        }
        
        with open('svm_metrics_summary_kfold.json', 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        
        print("\n" + "="*70)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*70)
        print("\nArchivos generados:")
        print("✓ svm_best_model_kfold.pkl - Modelo entrenado")
        print("✓ svm_metrics_summary_kfold.json - Resumen de métricas")
        print("✓ svm_kfold_analysis.png - Análisis de K-Fold")
        print("✓ svm_confusion_matrix_detailed.png - Matriz de confusión")
        print("✓ svm_roc_curves.png - Curvas ROC")
        print("✓ svm_feature_importance.png - Importancia de características")
        print("✓ svm_probability_calibration.png - Calibración de probabilidades")
        print("✓ svm_comprehensive_report.png - Reporte completo")
        print("="*70)
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()