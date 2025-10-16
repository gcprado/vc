#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Microplásticos - Versión Simple
Clasificación basada en reglas usando características morfológicas
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def get_config():
    """
    Retorna la configuración de rutas y parámetros del proyecto.
    """
    config = {
        "image_paths": {
            "FRA": [
                "./assets/microplasticos/FRA.png",
                "./assets/microplasticos/fragment-03-olympus-10-01-2020.JPG"
            ],
            "PEL": [
                "./assets/microplasticos/PEL.png",
                "./assets/microplasticos/pellet-03-olympus-10-01-2020.JPG"
            ],
            "TAR": [
                "./assets/microplasticos/TAR.png",
                "./assets/microplasticos/tar-03-olympus-10-01-2020.JPG"
            ]
        },
        "output_dir": "./assets/microplasticos/augmentation",
        "test_image_path": "./assets/microplasticos/MPs_test.jpg",
        "test_csv_path": "./assets/microplasticos/MPs_test_bbs.csv",
        "augmentation_params": {
            "rotation_angles": [90, 180, 270],
            "brightness_factors": [0.8, 1.2],
            "noise_levels": [10],
            "blur_kernel_sizes": [3]
        },
        "classification_params": {
            "circularity_threshold": 0.785,
            "value_threshold": 182.5,
            "min_area": 100
        }
    }
    return config


# ============================================================================
# AUGMENTACIÓN DE IMÁGENES
# ============================================================================

def apply_rotation(image, angle):
    """Rota una imagen por el ángulo especificado."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def apply_brightness(image, factor):
    """Ajusta el brillo de una imagen."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_noise(image, level):
    """Añade ruido gaussiano a una imagen."""
    noise = np.random.normal(0, level, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy


def apply_blur(image, kernel_size):
    """Aplica desenfoque gaussiano."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def generate_augmented_images(image, augmentation_params):
    """
    Genera múltiples versiones aumentadas de una imagen.
    
    Args:
        image: Imagen original
        augmentation_params: Diccionario con parámetros de augmentación
    
    Returns:
        Lista de tuplas (nombre_transformación, imagen_aumentada)
    """
    augmented = [("original", image.copy())]
    
    # Flips
    augmented.append(("flip_horizontal", cv2.flip(image, 1)))
    augmented.append(("flip_vertical", cv2.flip(image, 0)))
    
    # Rotaciones
    for angle in augmentation_params.get("rotation_angles", []):
        augmented.append((f"rotate_{angle}", apply_rotation(image, angle)))
    
    # Brillo
    for idx, factor in enumerate(augmentation_params.get("brightness_factors", [])):
        brightness_type = "bright" if factor > 1.0 else "dark"
        augmented.append((f"{brightness_type}_{idx:02d}", apply_brightness(image, factor)))
    
    # Ruido
    for idx, level in enumerate(augmentation_params.get("noise_levels", [])):
        augmented.append((f"noise_{idx:02d}", apply_noise(image, level)))
    
    # Desenfoque
    for idx, kernel_size in enumerate(augmentation_params.get("blur_kernel_sizes", [])):
        augmented.append((f"blur_{idx:02d}", apply_blur(image, kernel_size)))
    
    return augmented


def generate_training_dataset(config):
    """
    Genera el conjunto de imágenes aumentadas para entrenamiento.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Diccionario con estadísticas de generación
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"total_images": 0, "by_label": {}}
    
    for label, paths in config["image_paths"].items():
        label_lower = label.lower()
        img_counter = 0
        
        for img_idx, img_path in enumerate(paths, start=1):
            img = cv2.imread(img_path)
            
            if img is not None:
                augmented_images = generate_augmented_images(img, config["augmentation_params"])
                
                for aug_name, aug_img in augmented_images:
                    img_counter += 1
                    filename = f"{label_lower}_{aug_name}_{img_idx:03d}.png"
                    out_path = output_dir / filename
                    cv2.imwrite(str(out_path), aug_img)
                
                stats["by_label"][label] = img_counter
                stats["total_images"] += img_counter
    
    print(f"✅ Generadas {stats['total_images']} imágenes aumentadas")
    for label, count in stats["by_label"].items():
        print(f"   - {label}: {count} imágenes")
    
    return stats


# ============================================================================
# VISUALIZACIÓN DE IMÁGENES DE ENTRENAMIENTO
# ============================================================================

def visualize_training_images(config, max_per_label=8):
    """
    Muestra una muestra de las imágenes de entrenamiento aumentadas.
    
    Args:
        config: Diccionario de configuración
        max_per_label: Número máximo de imágenes a mostrar por clase
    """
    output_dir = Path(config["output_dir"])
    
    fig, axes = plt.subplots(len(config["image_paths"]), max_per_label, 
                             figsize=(20, 8))
    fig.suptitle("Muestra de Imágenes de Entrenamiento Aumentadas", 
                 fontsize=20, fontweight='bold')
    
    for i, label in enumerate(config["image_paths"].keys()):
        label_lower = label.lower()
        image_files = sorted(output_dir.glob(f"{label_lower}_*.png"))[:max_per_label]
        
        for j, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(img_rgb)
            axes[i, j].set_title(f"{label}\n{img_path.name}", fontsize=8)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================================

def extract_features(image, min_area=100):
    """
    Extrae características de una imagen de microplástico.
    
    Args:
        image: Imagen en formato BGR
        min_area: Área mínima para considerar un contorno válido
    
    Returns:
        Tupla (mean_val, mean_weighted_circularity, valid_contours)
    """
    if image is None or image.size == 0:
        return 0, 0, []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_val = np.mean(hsv[:, :, 2])

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_weighted_circularity = 0
    total_area = 0
    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if area > min_area:
            valid_contours.append(c)
            total_area += area

            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                total_weighted_circularity += circularity * area

    if total_area > 0:
        mean_weighted_circularity = total_weighted_circularity / total_area
    else:
        mean_weighted_circularity = 0

    return mean_val, mean_weighted_circularity, valid_contours


def visualize_feature_extraction(config):
    """
    Visualiza la extracción de características sobre las imágenes originales.
    
    Args:
        config: Diccionario de configuración
    """
    image_paths = config["image_paths"]
    min_area = config["classification_params"]["min_area"]
    
    fig, axes = plt.subplots(1, len(image_paths), figsize=(18, 5))
    fig.suptitle("Detección de Objetos (OTSU + Filtro de Área)", 
                 fontsize=24, fontweight='bold')

    for i, (label, paths) in enumerate(image_paths.items()):
        path = paths[0]
        full_path = Path(path)

        if full_path.exists():
            img = cv2.imread(str(full_path))
            if img is not None:
                mean_val, circularity, valid_contours = extract_features(img, min_area)

                vis_img = img.copy()

                if valid_contours:
                    cv2.drawContours(vis_img, valid_contours, -1, (0, 255, 255), 3)

                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

                axes[i].imshow(vis_img_rgb)
                axes[i].set_title(f"{label}\n(Circ: {circularity:.2f}, Val: {mean_val:.1f})", 
                                fontsize=14)
                axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================================
# CLASIFICACIÓN Y PREDICCIÓN
# ============================================================================

def predict_particle_class(image, config):
    """
    Predice la clase de una partícula usando reglas basadas en características.
    
    Args:
        image: Imagen de la partícula en formato BGR
        config: Diccionario de configuración
    
    Returns:
        String con la etiqueta predicha ("FRA", "PEL", o "TAR")
    """
    params = config["classification_params"]
    mean_val, circularity, _ = extract_features(image, params["min_area"])

    if circularity > params["circularity_threshold"]:
        return "PEL"
    elif mean_val <= params["value_threshold"]:
        return "TAR"
    else:
        return "FRA"


# ============================================================================
# ANÁLISIS DE CARACTERÍSTICAS DEL CONJUNTO DE ENTRENAMIENTO
# ============================================================================

def build_training_features_dataframe(config):
    """
    Construye un DataFrame con las características de todas las imágenes de entrenamiento.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        DataFrame con características extraídas
    """
    output_dir = config["output_dir"]
    min_area = config["classification_params"]["min_area"]
    features_data = []

    for label in config["image_paths"].keys():
        label_lower = label.lower()
        image_files = sorted(Path(output_dir).glob(f"{label_lower}_*.png"))

        for file_path in image_files:
            img = cv2.imread(str(file_path))

            if img is not None:
                mean_val, circularity, _ = extract_features(img, min_area)

                features_data.append({
                    'FileName': file_path.name,
                    'Label': label,
                    'Mean_Val_HSV': mean_val,
                    'Circularity_Weighted': circularity
                })

    return pd.DataFrame(features_data)


def display_training_features(features_df, max_rows=20):
    """
    Muestra el DataFrame de características del conjunto de entrenamiento.
    
    Args:
        features_df: DataFrame con características
        max_rows: Número máximo de filas a mostrar (None para todas)
    """
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.width', 1000)

    print("\n" + "=" * 20 + " CARACTERÍSTICAS DEL CONJUNTO DE ENTRENAMIENTO " + "=" * 20)
    print(features_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)
    
    print(f"\n📊 Resumen estadístico por clase:")
    print(features_df.groupby('Label')[['Mean_Val_HSV', 'Circularity_Weighted']].describe())

    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')


# ============================================================================
# EVALUACIÓN EN CONJUNTO DE TEST
# ============================================================================

def evaluate_test_set(config):
    """
    Evalúa el clasificador en el conjunto de test.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Tupla (accuracy, confusion_matrix, results_df, y_true, y_pred)
    """
    test_img = cv2.imread(config["test_image_path"])
    bbs = pd.read_csv(config["test_csv_path"])
    min_area = config["classification_params"]["min_area"]
    
    y_true, y_pred = [], []
    df_data = []

    for idx, row in bbs.iterrows():
        x1, y1, x2, y2 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
        label = row['label'].strip().upper()

        y1, y2 = max(0, y1), min(test_img.shape[0], y2)
        x1, x2 = max(0, x1), min(test_img.shape[1], x2)

        roi = test_img[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] <= 1 or roi.shape[1] <= 1:
            continue

        mean_val, circularity, _ = extract_features(roi, min_area)
        pred = predict_particle_class(roi, config)

        y_true.append(label)
        y_pred.append(pred)

        df_data.append({
            'Index_ROI': idx,
            'True_Label': label,
            'Pred_Label': pred,
            'Mean_Val_HSV': mean_val,
            'Circularity_Weighted': circularity
        })

    target_labels = ["FRA", "PEL", "TAR"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=target_labels)
    results_df = pd.DataFrame(df_data)

    return acc, cm, results_df, y_true, y_pred


def display_evaluation_results(acc, results_df, max_rows=10):
    """
    Muestra los resultados de la evaluación.
    
    Args:
        acc: Accuracy del clasificador
        results_df: DataFrame con predicciones
        max_rows: Número de filas a mostrar
    """
    print(f"\n{'='*30}")
    print(f"📈 Accuracy: {acc * 100:.2f}%")
    print(f"{'='*30}")

    print("\n" + "=" * 20 + " PREDICCIONES EN CONJUNTO DE TEST (Muestra) " + "=" * 20)
    print(results_df.head(max_rows).to_string(index=False, float_format="%.3f"))
    print("=" * 85)


# ============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================================

def visualize_predictions(config, y_pred):
    """
    Visualiza las predicciones sobre la imagen de test.
    
    Args:
        config: Diccionario de configuración
        y_pred: Lista de predicciones
    """
    test_img = cv2.imread(config["test_image_path"])
    bbs = pd.read_csv(config["test_csv_path"])
    
    vis_img = test_img.copy()
    
    colors = {
        "FRA": (255, 0, 0),    # Azul
        "PEL": (0, 255, 0),    # Verde
        "TAR": (0, 0, 255)     # Rojo
    }
    
    for idx, row in bbs.iterrows():
        if idx >= len(y_pred):
            break
            
        x1, y1 = int(row['x_min']), int(row['y_min'])
        x2, y2 = int(row['x_max']), int(row['y_max'])
        pred = y_pred[idx]
        
        color = colors.get(pred, (255, 255, 255))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, pred, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Predicciones sobre Imagen de Test", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(cm):
    """
    Visualiza la matriz de confusión.
    
    Args:
        cm: Matriz de confusión
    """
    labels = ["FRA", "PEL", "TAR"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Número de muestras'})
    plt.title("Matriz de Confusión", fontsize=16, fontweight='bold')
    plt.ylabel("Etiqueta Real", fontsize=12)
    plt.xlabel("Etiqueta Predicha", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJECUCIÓN PRINCIPAL (para testing o ejecución directa)
# ============================================================================

if __name__ == "__main__":
    # 1. Obtener configuración
    print("📋 Cargando configuración...")
    config = get_config()
    
    # 2. Generar dataset de entrenamiento aumentado
    print("\n🔄 Generando imágenes aumentadas...")
    stats = generate_training_dataset(config)
    
    # 3. Visualizar muestra de imágenes de entrenamiento
    print("\n📊 Visualizando muestra de imágenes de entrenamiento...")
    visualize_training_images(config, max_per_label=8)
    
    # 4. Visualizar extracción de características
    print("\n🔍 Visualizando extracción de características...")
    visualize_feature_extraction(config)
    
    # 5. Construir DataFrame de características de entrenamiento
    print("\n📈 Construyendo DataFrame de características...")
    features_df = build_training_features_dataframe(config)
    display_training_features(features_df, max_rows=None)
    
    # 6. Evaluar en conjunto de test
    print("\n🎯 Evaluando en conjunto de test...")
    acc, cm, results_df, y_true, y_pred = evaluate_test_set(config)
    display_evaluation_results(acc, results_df, max_rows=10)
    
    # 7. Visualizar predicciones
    print("\n🖼️ Visualizando predicciones...")
    visualize_predictions(config, y_pred)
    
    # 8. Visualizar matriz de confusión
    print("\n📉 Visualizando matriz de confusión...")
    visualize_confusion_matrix(cm)
