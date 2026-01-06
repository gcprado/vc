# Un estudio comparativo sobre YOLOv8 y SAM+LVM aplicado a un Sistema de Detección, Conteo y Gestión de Inventario 

## Descripción del Proyecto

Este proyecto desarrolla un sistema de visión por computador capaz de:

Parte teorica:
- Evaluar la precisión y consistencia de un modelo YOLO + LVM en entornos de inventario realistas.
- Evaluar la precision y consistencia de un modelo SAM + LVM en entornos de inventario realistas.
- Comparar el rendimiento de dos enfoques: **YOLOv8 entrenado** vs **SAM + LVM**.

Parte practica:
- Detectar, contar y seguir productos en **estanterías**.
- Detectar reposiciones y retiros de productos.
- Generar un panel de control de **inventario** en tiempo real.
- Generar gráficos de evolución de inventario por producto y realizar predicciones sobre stock.
- Analizar resutaldos con LLM.
- Exportar ficheros csv con resultaods

---

### Dataset y Entrenamiento del Modelo

#### Dataset SKU110K
El dataset SKU110K es un conjunto de imágenes de estanterías comerciales con alta densidad de productos, desarrollado para tareas de detección y conteo de objetos en entornos de retail. Contiene más de 11,000 imágenes y alrededor de 400,000 instancias de productos, etiquetadas mediante bounding boxes. Las imágenes incluyen múltiples desafíos típicos de inventario real, como superposición de productos, variaciones de iluminación, perspectivas diferentes, y oclusiones parciales, lo que lo convierte en un dataset adecuado para evaluar modelos de detección y segmentación en entornos logísticos.

El dataset SKU110K se organizó siguiendo la estructura estándar de YOLO:

```
.
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
```

**Training set:** Este subconjunto contiene 8,219 imágenes y sus anotaciones, utilizadas para entrenar los modelos de detección de objetos.  

**Validation set:** Este subconjunto consta de 588 imágenes y anotaciones, usadas para la validación del modelo durante el entrenamiento.  

**Test set:** Este subconjunto incluye 2,936 imágenes, diseñadas para la evaluación final de los modelos entrenados de detección de objetos.

**Archivo YAML oficial para el entrenamiento con Ultralytics:** [Ver aquí](https://docs.ultralytics.com/datasets/detect/sku-110k/#dataset-yaml)  
**Artículo original del dataset SKU110K:** [Leer Paper](https://arxiv.org/pdf/1904.00853)

#### Modelo y entrenamiento
Se utilizó un modelo base **YOLOv8n** preentrenado en COCO, adaptado al dataset.  
El entrenamiento se realizó durante 10 épocas con imágenes de 640×640 píxeles y batch size de 8. 

El modelo resultante se guarda en:
`yolov8_train_results/content/runs/detect/train/weights/best.pt`

---

### Resultados del Entrenamiento

Durante el proceso de entrenamiento se generaron diversas gráficas que permiten analizar el rendimiento del modelo. Todas se encuentran en: `yolov8_train_results/content/runs/detect/train/`

A continuación se incluyen los enlaces a las principales métricas visuales:

| Métrica | Imagen |
|----------|--------|
| F1-Score vs Confidence | ![BoxF1_curve](yolov8_train_results/content/runs/detect/train/BoxF1_curve.png) |
| Precision vs Confidence | ![BoxP_curve](yolov8_train_results/content/runs/detect/train/BoxP_curve.png) |
| Precision vs Recall | ![BoxPR_curve](yolov8_train_results/content/runs/detect/train/BoxPR_curve.png) |
| Recall vs Confidence | ![BoxR_curve](yolov8_train_results/content/runs/detect/train/BoxR_curve.png) |
| Confusion Matrix | ![confusion_matrix](yolov8_train_results/content/runs/detect/train/confusion_matrix.png) |
| Confusion Matrix (Normalized) | ![confusion_matrix_normalized](yolov8_train_results/content/runs/detect/train/confusion_matrix_normalized.png) |
| Labels | ![labels](yolov8_train_results/content/runs/detect/train/labels.jpg) |
| Other metrics | ![results](yolov8_train_results/content/runs/detect/train/results.png) |

A continuación se muestran ejemplos del conjunto de entrenamiento y validación, generados automáticamente por YOLO:

| Tipo | Imagen |
|------|---------|
| Ejemplo de batch de entrenamiento (1) | ![train_batch0](yolov8_train_results/content/runs/detect/train/train_batch0.jpg) |
| Ejemplo de batch de entrenamiento (2) | ![train_batch1](yolov8_train_results/content/runs/detect/train/train_batch1.jpg) |
| Ejemplo de batch de entrenamiento (3) | ![train_batch2](yolov8_train_results/content/runs/detect/train/train_batch2.jpg) |
| Etiquetas del conjunto de validación | ![val_batch0_labels](yolov8_train_results/content/runs/detect/train/val_batch0_labels.jpg) |
| Predicciones del conjunto de validación | ![val_batch0_pred](yolov8_train_results/content/runs/detect/train/val_batch0_pred.jpg) |

---

### Evaluación del Modelo

La validación se realizó sobre el conjunto de prueba (split *test*), utilizando el modelo **YOLOv8n** entrenado durante 10 épocas con imágenes de 640×640 píxeles en batches de tamaño 8.

**Resumen del rendimiento final:**

| Métrica | Valor |
|---------|-------|
| Precisión (Box Accuracy) | 88.3% |
| Sensibilidad (Recall) | 80.4% |
| Precisión media (mAP@50) | 87.8% |
| Precisión media (mAP@50–95) | 54.9% |
| Box Loss final | 1.395 |
| Class Loss final | 0.662 |
| DFL Loss final | 0.998 |
| GPU utilizada | Tesla T4 (15GB) |
| Parámetros del modelo | 3,005,843 |
| FLOPs | 8.1 GFLOPs |

**Notas adicionales:**
- El optimizador se seleccionó automáticamente (AdamW), ajustando `lr` y `momentum` de forma óptima.  
- Se aplicaron transformaciones de aumento de datos leves durante el entrenamiento, incluyendo blur, median blur, conversión a gris y CLAHE con baja probabilidad, para mejorar la generalización.  
- El modelo validado mostró consistencia entre las métricas de precisión y recall a lo largo de los 10 epochs, indicando buen balance entre detección de objetos y control de falsos positivos.

---

### Comparativa de Métodos

Se evaluaron dos métodos de reconocimiento de texto:

| Tipo de Métrica       | Descripción                                                    | YOLO             | SAM + LVM         |
|-----------------------|----------------------------------------------------------------|------------------|-------------------|
| **Total de Muestras** | Número de imagenes evaluadas                                   | **31**           | **31**            |
| **Deteccion Exacta**  | La cantidad predicha coincide exactamente con la cantidad real | **xx.x% (x/31)** | **xx.xx% (x/31)** |
| **Deteccion Parcial** | Porcentaje de deteccion sobre la cantidad real.                | **xx.xx%**       | **xx.xx%**        |


**Datos de evaluacion:** [Descargar desde Google Drive]()

**Conclusiones:**

---

### Posibles Extensiones

- Integración con **LLM's** para generar reportes automáticos y alertas detalladas.   
- Comparación con otros modelos de segmentación y detección para optimización del pipeline.  
- Desarrollo de aplicacion full-stack que integre todo.  
  
---

### Autor

**Giancarlo Prado Abreu**  
- Trabajo Práctico de Visión por Computador  
- Escuela de Ingeniería Informática - ULPGC
