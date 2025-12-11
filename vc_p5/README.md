# Clasificación de Edad con Transfer Learning y ResNet50. Aplicación de filtros con mediapipe

## Descripción del Proyecto

Este proyecto desarrolla un sistema de clasificación de edad facial utilizando técnicas de deep learning y transfer learning. El sistema es capaz de:

- Organizar y preprocesar el **dataset UTKFace** por categorías de edad.  
- Entrenar un modelo basado en **ResNet50** para clasificar rostros en tres grupos: **joven**, **medio** y **anciano**.  
- Aplicar **filtros personalizados** a las imágenes.

---

## Dataset

Se utilizó el dataset **UTKFace**, que contiene imágenes de rostros con anotaciones de edad, género y etnia. Para este proyecto:

- Las imágenes se organizaron en tres categorías de edad:
  - **Joven**: menores de 40 años
  - **Medio**: entre 40 y 64 años
  - **Anciano**: 65 años o más

- El dataset fue dividido estratificadamente en:
  - **70% entrenamiento** (7,098 imágenes)
  - **15% validación** (1,524 imágenes)
  - **15% test** (1,524 imágenes)

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
dataset_by_age
├── train
│   ├── joven
│   ├── medio
│   └── anciano
├── validation
│   ├── joven
│   ├── medio
│   └── anciano
└── test
    ├── joven
    ├── medio
    └── anciano
```

**Dataset de entrenamiento:** [Descargar desde Google Drive](https://drive.google.com/file/d/1EJhO_b12raN6XgT_f4VouoKXd3Bik1PM/view?usp=sharing)

---

### vc_p5b.ipynb - Entrenamiento del Modelo

Este cuaderno implementa el entrenamiento del modelo de clasificación:

**Arquitectura del modelo:**
- Base: **ResNet50** preentrenado en ImageNet (capas congeladas)
- Capa de pooling global
- Dropout (0.3) para evitar overfitting
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
- El modelo no ha aprendido lo suficiente, seria aconsejable descongelar mas capas.

**Fine-tuning:**
- Se descongelaron las últimas 50 capas de ResNet50
- Learning rate reducido a 1e-5
- Entrenamiento adicional por 5 épocas
- Da lugar a overfitting, aconsejable probar con menos capas.

---

### vc_p5c.ipynb - Aplicación de Filtros y Efectos Visuales

Este cuaderno implementa un sistema de aplicación de filtros y efectos visuales basado en la edad predicha basandose en el modelo entrenado.

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
   - Aplica superposiciones, máscaras o efectos según la edad detectada usando mediapipe.
   - Maneja casos donde los filtros no están disponibles

**Demostraciones visuales:**

![Clasificación de edad en tiempo real](out/age.gif)  
*Sistema de clasificación de edad detectando y categorizando rostros en las tres clases: joven, medio y anciano*

![Filtros de ojos animados](out/eyes.gif)  
*Aplicación de filtros AR (ojos animados) basados en la edad detectada*

---

## 👨‍💻 Autor

**Giancarlo Prado Abreu**  
- Práctica 5 de la asignatura Visión por Computador
- Escuela de Ingeniería Informática - ULPGC
