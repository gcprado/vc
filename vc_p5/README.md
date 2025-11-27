# Clasificación de Edad con Transfer Learning y ResNet50

## Descripción del Proyecto

Este proyecto desarrolla un sistema de clasificación de edad facial utilizando técnicas de deep learning y transfer learning. El sistema es capaz de:

- Organizar y preprocesar el **dataset UTKFace** por categorías de edad.  
- Entrenar un modelo basado en **ResNet50** para clasificar rostros en tres grupos: **joven**, **medio** y **anciano**.  
- Aplicar **filtros personalizados** a las imágenes según la edad predicha.  
- Generar datasets balanceados con igual cantidad de muestras por clase.  

---

## Dataset

Se utilizó el dataset **UTKFace**, que contiene más de 20,000 imágenes de rostros con anotaciones de edad, género y etnia. Para este proyecto:

- Las imágenes se organizaron en tres categorías de edad:
  - **Joven**: menores de 40 años
  - **Medio**: entre 40 y 64 años
  - **Anciano**: 65 años o más

- El dataset fue dividido estratificadamente en:
  - **70% entrenamiento** (7,095 imágenes)
  - **15% validación** (1,521 imágenes)
  - **15% test** (1,521 imágenes)

**Preprocesamiento:**
- Todas las imágenes fueron redimensionadas a **128×128 píxeles**.
- Se generaron nombres únicos con UUID para evitar conflictos.
- Se creó una versión normalizada del dataset con igual cantidad de muestras por clase (1,148 imágenes por categoría).

---

## Cuadernos de Trabajo

### vc_p5a.ipynb - Preparación del Dataset

Este cuaderno se encarga de organizar el dataset UTKFace:

- **Extracción de edad** desde los nombres de archivo (formato: `edad_genero_etnia_timestamp.jpg`)
- **Categorización** en tres grupos de edad (joven, medio, anciano)
- **División estratificada** en train/validation/test (70/15/15)
- **Normalización del dataset** para balancear las clases

**Estructura de salida:**
```
dataset_by_age2/
├── train/
│   ├── joven/
│   ├── medio/
│   └── anciano/
├── validation/
│   ├── joven/
│   ├── medio/
│   └── anciano/
└── test/
    ├── joven/
    ├── medio/
    └── anciano/
```

---

### vc_p5b.ipynb - Entrenamiento del Modelo

Este cuaderno implementa el entrenamiento del modelo de clasificación:

**Arquitectura del modelo:**
- Base: **ResNet50** preentrenado en ImageNet (capas congeladas)
- Capa de pooling global
- Dropout (0.3) para regularización
- Capa densa de salida con 3 clases (softmax)

**Configuración del entrenamiento:**
- Optimizador: Adam
- Función de pérdida: Categorical Crossentropy
- Tamaño de batch: 16
- Épocas máximas: 30
- Early stopping con paciencia de 10 épocas

**Resultados del entrenamiento:**
- El modelo se detuvo automáticamente en la época 12
- **Precisión en test: 72.65%**
- Se guardó el modelo como `model_age_classification.keras`

**Fine-tuning:**
- Se descongelaron las últimas 50 capas de ResNet50
- Learning rate reducido a 1e-5
- Entrenamiento adicional por 5 épocas

---

### vc_p5c.ipynb - Aplicación de Filtros y Efectos Visuales

Este cuaderno implementa un sistema completo de aplicación de filtros y efectos visuales basado en la edad predicha. Es la culminación práctica del proyecto, donde el modelo entrenado se utiliza para crear experiencias interactivas.

**Pipeline de procesamiento:**

1. **Carga del modelo:** Importa el modelo entrenado `model_age_classification.keras` (230 MB) con todos sus pesos
2. **Preprocesamiento de imágenes:** 
   - Redimensiona las imágenes a 128×128 píxeles
   - Normaliza los valores de píxeles
   - Prepara los datos para la inferencia
3. **Inferencia del modelo:**
   - Obtiene probabilidades para las tres clases (anciano, joven, medio)
   - Identifica la categoría con mayor confianza
   - Registra todas las probabilidades para análisis
4. **Sistema de filtros:**
   - Busca filtros específicos en `out/filters/` para cada categoría
   - Aplica superposiciones, máscaras o efectos según la edad detectada
   - Maneja casos donde los filtros no están disponibles

**Características técnicas:**

- **Debug mode:** Muestra las probabilidades completas de cada predicción para verificación
- **Salida formateada:** 
  ```
  → [filename] | PROBS=[p1, p2, p3] | predicción=[clase]
  ```
- **Gestión de archivos:** Los filtros se organizan por categoría en carpetas separadas
- **Visualización:** Muestra imágenes originales con predicciones y confianza

**Ejemplo de salida:**
```
DEBUG – Probabilidades completas: [0.01846885 0.80786073 0.1736704]
→ imagen.png | PROBS=[0.018 0.808 0.174] | predicción=joven
```

**Demostraciones visuales:**

Los siguientes GIFs muestran el sistema en acción, demostrando diferentes aspectos de la clasificación de edad y aplicación de filtros:

![Clasificación de edad en tiempo real](out/age.gif)  
*Sistema de clasificación de edad detectando y categorizando rostros en las tres clases: joven, medio y anciano*

![Filtros de ojos animados](out/eyes.gif)  
*Aplicación de filtros AR (ojos animados) basados en la edad detectada*

Estos ejemplos demuestran la capacidad del modelo para:
- Clasificar correctamente rostros en diferentes grupos de edad
- Aplicar filtros y efectos visuales de forma dinámica
- Funcionar en tiempo real con diferentes tipos de imágenes
- Manejar variaciones en iluminación, ángulos y expresiones faciales

**Aplicaciones prácticas:**
- **Filtros AR:** Efectos de realidad aumentada basados en edad
- **Análisis demográfico:** Clasificación automática en aplicaciones comerciales
- **Control de contenido:** Restricción de acceso según grupo etario
- **Entretenimiento:** Filtros divertidos para redes sociales
- **Investigación:** Análisis de distribución de edades en datasets

---

## Resultados

### Métricas del Modelo

- **Precisión de entrenamiento final:** ~71%
- **Precisión de validación final:** ~72%
- **Precisión de test:** **72.65%**

El modelo muestra un buen balance entre entrenamiento y validación, sin signos evidentes de sobreajuste gracias al uso de:
- Transfer learning con ResNet50 preentrenado
- Dropout para regularización
- Early stopping para evitar sobreentrenamiento

### Archivos Generados

- **Modelo entrenado:** `model_age_classification.keras` (230 MB)
- **Datasets organizados:**
  - `dataset_by_age2/` (dataset completo)
  - `dataset_by_age_normalized2/` (dataset balanceado)
- **Gráficas de entrenamiento:** Accuracy y Loss por época

---

## Posibles Extensiones

- Aumentar el número de categorías de edad para clasificación más fina.
- Implementar data augmentation para mejorar la generalización.
- Explorar otras arquitecturas (EfficientNet, Vision Transformer).
- Desarrollar una aplicación web interactiva para clasificación en tiempo real.
- Agregar detección de rostros previa para procesamiento automático de imágenes.
- Implementar filtros AR más sofisticados según la edad detectada.

---

## 👨‍💻 Autor

**Giancarlo Prado Abreu**  
- Práctica 5 de la asignatura Visión por Computador
- Escuela de Ingeniería Informática - ULPGC
