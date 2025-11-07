# Detección, Seguimiento y Reconocimiento de Matrículas con YOLO, EasyOCR y PaddleOCR

## Descripción del Proyecto

Este proyecto desarrolla un sistema de visión por computadora capaz de:

- Detectar y seguir **personas** y **vehículos** en vídeos.  
- Detectar y reconocer las **matrículas** de los vehículos.  
- Generar un **vídeo anotado** con las detecciones visualizadas.  
- Exportar un **archivo CSV** con los resultados.  
- Comparar el rendimiento de dos OCR: **EasyOCR** y **PaddleOCR**.  
- Evaluar la precisión de un modelo YOLO personalizado entrenado para la detección de matrículas.

---

## Entrenamiento del Modelo

Se utilizó un modelo base **YOLOv11-Large** preentrenado, adaptado a un conjunto de datos específico de matrículas.  
El entrenamiento se realizó durante 150 épocas con imágenes de 640×640 píxeles, aplicando *early stopping* para evitar sobreajuste.

El modelo resultante se guarda en:  
`matriculas_model/version_1/weights/best.pt`

El conjunto de datos utilizado para entrenar el modelo YOLOv11 fue preparado siguiendo la estructura estándar, separando los conjuntos de **entrenamiento**, **validación** y **prueba**, cada uno con sus respectivas carpetas de imágenes y etiquetas en formato `.txt` (YOLO format):

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

**Dataset de entrenamiento:** [Descargar desde Google Drive](https://drive.google.com/file/d/1F463kIUb08GTNUCcK7W4LyZajuUZ65lW/view?usp=sharing)

---

## Resultados del Entrenamiento

Durante el proceso de entrenamiento se generaron diversas gráficas que permiten analizar el rendimiento del modelo. Todas se encuentran en: `matriculas_model/version_1/`


A continuación se incluyen los enlaces a las principales métricas visuales:

| Métrica | Imagen |
|----------|--------|
| F1-Score vs Confidence | ![BoxF1_curve](matriculas_model/version_1/BoxF1_curve.png) |
| Precision vs Confidence | ![BoxP_curve](matriculas_model/version_1/BoxP_curve.png) |
| Precision vs Recall | ![BoxPR_curve](matriculas_model/version_1/BoxPR_curve.png) |
| Recall vs Confidence | ![BoxR_curve](matriculas_model/version_1/BoxR_curve.png) |
| Confusion Matrix | ![confusion_matrix](matriculas_model/version_1/confusion_matrix.png) |
| Confusion Matrix (Normalized) | ![confusion_matrix_normalized](matriculas_model/version_1/confusion_matrix_normalized.png) |
| Labels | ![labels](matriculas_model/version_1/labels.jpg) |
| Other metrics | ![results](matriculas_model/version_1/results.png) |

A continuación se muestran ejemplos del conjunto de entrenamiento y validación, generados automáticamente por YOLO:

| Tipo | Imagen |
|------|---------|
| Ejemplo de batch de entrenamiento (1) | ![train_batch0](matriculas_model/version_1/train_batch0.jpg) |
| Ejemplo de batch de entrenamiento (2) | ![train_batch1](matriculas_model/version_1/train_batch1.jpg) |
| Ejemplo de batch de entrenamiento (3) | ![train_batch2](matriculas_model/version_1/train_batch2.jpg) |
| Etiquetas del conjunto de validación | ![val_batch0_labels](matriculas_model/version_1/val_batch0_labels.jpg) |
| Predicciones del conjunto de validación | ![val_batch0_pred](matriculas_model/version_1/val_batch0_pred.jpg) |

---

## Evaluación del Modelo

La validación se llevó a cabo sobre el conjunto de prueba (split *test*) definido en `data.yaml`.

**Resultados destacados:**
- Precisión (Accuracy): 93.2%  
- Sensibilidad (Recall): 76.3%  
- Precisión media (mAP@50): 81.0%  
- Precisión media (mAP@50–95): 54.9%  

---

## Procesamiento del Vídeo

El sistema fue probado sobre un vídeo de ejemplo, realizando las siguientes tareas:

- Detección de **personas** y **vehículos** utilizando el modelo COCO base (`yolo11l.pt`).  
- Detección de **matrículas** con el modelo personalizado (`best.pt`).  
- Reconocimiento de texto mediante **EasyOCR** y **PaddleOCR**.  
- Seguimiento simple por centroides para mantener identificadores entre fotogramas.  
- Generación de un **vídeo con anotaciones** y un **archivo CSV** con todas las detecciones.

---

## Resultados

- **Vídeo original:** [C0142.MP4](https://drive.google.com/file/d/1aY4ROz7G3PcyhdQZp1BRLN6NohsX8mlF/view?usp=sharing)  
- **Vídeo procesado (resultados):** [detecciones_y_ocr.mp4](https://drive.google.com/file/d/1Dq_CaNwxfpyMGIyga36OlxNeadNXorfp/view?usp=sharing
- **Archivo CSV generado:** [out/reporte_ocr_final.csv](out/reporte_ocr_final.csv)
- **Archivo CSV con matriculas filtradas:** [out/matriculas_filtradas_final.csv](out/matriculas_filtradas_final.csv)  

El archivo `reporte_ocr_final.csv` incluye, para cada detección:  
número de fotograma, tipo de objeto, confianza, identificador de tracking, coordenadas de la caja delimitadora, matrícula detectada, coordenadas de la matrícula y resultados OCR con sus respectivas confianzas.

El archivo `matriculas_filtradas_final.csv` fue generado mediante un proceso de post-filtrado que:
- Convierte el número de fotograma en tiempo (minutos:segundos).  
- Filtra las detecciones de **PaddleOCR** con confianza superior a `0.60`.  
- Valida el formato de matrícula española (`4 dígitos + 3 letras`).  
- Conserva únicamente la detección más confiable para cada matrícula.  

Este filtrado permite obtener un listado limpio y preciso de las matrículas detectadas, ideal para reportes o análisis posteriores.

---

## Comparativa de OCR (FALTA POR TERMINAR)

Se evaluaron dos métodos de reconocimiento de texto:

| Método | Precisión Media | Tiempo Promedio por Detección (ms) | Robustez ante Ruido |
|---------|------------------|------------------------------------|----------------------|
| EasyOCR | xx.x% | xx.x | Media |
| PaddleOCR | xx.x% | xx.x | Alta |

**Conclusiones OCR:**
- PaddleOCR mostró mejor rendimiento frente a matrículas borrosas, sucias o inclinadas.  
- EasyOCR fue más rápido, ideal para aplicaciones en tiempo real.  
- Para análisis offline y mayor precisión, se recomienda PaddleOCR.

---

## Posibles Extensiones

- Análisis del flujo direccional de vehículos y personas (entradas/salidas).  
- Detección de matrículas mediante métodos basados en contornos.  
- Anonimización automática de personas o matrículas mediante desenfoque.  
- Implementación de un *tracker* avanzado (DeepSORT, ByteTrack, etc.).
  
---

## 👨‍💻 Autor

**Giancarlo Prado Abreu**  
- Práctica 4 de la asignatura Visión por Computador
- Escuela de Ingeniería Informática - ULPGC
