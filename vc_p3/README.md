# VC_P3 - Visión por Computador: Detección de Monedas y Clasificación de Microplásticos

Proyecto 3 de Visión por Computador que implementa dos sistemas principales:
1. Sistema de detección, clasificación y valoración automática de monedas de euro
2. Sistema de clasificación de microplásticos mediante características geométricas y de apariencia

## 📋 Descripción General

Este proyecto aborda dos desafíos principales de visión por computador:

### Parte 1: Identificación de Monedas
1. **Detección Automática de Monedas**: Identificación de monedas en imágenes mediante múltiples algoritmos.
2. **Calibración Interactiva**: Sistema de calibración manual para establecer la escala píxeles-milímetros.
3. **Clasificación por Tamaño**: Identificación del valor de cada moneda basándose en su diámetro real.
4. **Cálculo de Valor Total**: Suma automática del valor total detectado.

### Parte 2: Clasificación de Microplásticos
1. **Augmentación de Datos**: Generación de variantes de imágenes para entrenamiento en caso de implementar un algoritmo de clasificacion ej: Random Forest.
2. **Extracción de Características**: Cálculo de brillo y circularidad ponderada.
3. **Clasificación Heurística**: Sistema basado en reglas sobre las caracteristicas extraidas para identificar tres tipos de microplásticos.
4. **Evaluación con Métricas**: Matriz de confusión y accuracy sobre conjunto de prueba.

---

## 🚀 Características Principales

### 1. Detección de Monedas

El sistema ofrece tres métodos de detección configurables:

**Método Avanzado por Contornos (contours_advanced):**
- Ecualización adaptativa de histograma (CLAHE) para uniformar iluminación
- Filtro bilateral para preservación de bordes
- Umbralización adaptativa Gaussiana
- Operaciones morfológicas (apertura y cierre) para limpiar ruido
- Filtrado por circularidad (≥0.55) y solidez (≥0.6)
- Ajuste de círculo mediante método de Kasa (mínimos cuadrados algebraicos)
- Refinamiento de radio usando mediana de distancias al contorno

**Método Hough Transform:**
- Detección de círculos mediante transformada de Hough
- Filtro de mediana para reducción de ruido
- Parámetros configurables (dp, minDist, param1, param2, minRadius, maxRadius)

**Método Contornos Básico (contours):**
- Umbralización global binaria
- Detección de contornos externos
- Filtrado por área mínima y circularidad

**Métricas de Calidad:**
- Circularidad: 4π·área / perímetro²
- Solidez: área / área del hull convexo
- Radio mediano para robustez ante oclusiones

---

### 2. Calibración Interactiva

Sistema de calibración manual mediante interfaz gráfica:

**Proceso:**
1. Visualización de todas las monedas detectadas con círculos amarillos
2. Selección interactiva de una moneda de referencia conocida (ej: 1€)
3. Cálculo automático del factor de conversión píxeles/mm
4. Resaltado de la moneda de referencia en verde

**Dimensiones de Referencia (Monedas de Euro):**
- 2.00€: 25.75 mm
- 1.00€: 23.25 mm
- 0.50€: 24.25 mm
- 0.20€: 22.25 mm
- 0.10€: 19.75 mm
- 0.05€: 21.25 mm
- 0.02€: 18.75 mm
- 0.01€: 16.25 mm

---

### 3. Clasificación de Monedas con Niveles de Confianza

Sistema de clasificación basado en diámetro real con rangos de tolerancia:

**Algoritmo de Clasificación:**
- Conversión de diámetro en píxeles a milímetros usando factor de calibración
- Definición de rangos: [nominal - tolerancia, nominal + tolerancia]
- Tolerancia configurable (por defecto: ±0.8 mm)

**Niveles de Confianza:**
- **Alta (High)**: Coincidencia única dentro del rango de tolerancia
- **Media (Medium)**: Múltiples coincidencias, se elige la más cercana
- **Baja (Low)**: Fuera de rango pero dentro de ±1.5 mm del valor nominal
- **Desconocida (Unknown)**: Error > 1.5 mm respecto al valor más cercano

**Criterios de Validación:**
- Cálculo de error absoluto en milímetros
- Selección del valor nominal más próximo
- Marcado de monedas no identificables

---

### 4. Visualización de Resultados

**Vista de Imagen Anotada:**
- Círculos de colores según tipo de moneda:
  - Dorado: Monedas de 2€, 1€, 0.50€, 0.20€, 0.10€
  - Cobrizo: Monedas de 0.05€, 0.02€, 0.01€
  - Gris: Monedas no identificadas
- Intensidad del color según confianza (más apagado = menor confianza)
- Grosor del círculo proporcional a la confianza
- Etiqueta con valor en euros sobre fondo negro

**Tabla Resumen:**
- Listado por denominación con cantidad y subtotal
- Suma total de todas las monedas identificadas

**Salida en Consola:**
- Resumen detallado con valor, cantidad, diámetros medidos y subtotales
- Listado de monedas no identificadas con sus diámetros
- Estadísticas de confianza (alta/media/baja/desconocida)
- Total de monedas identificadas vs detectadas

---

## 🧪 Sistema de Clasificación de Microplásticos

### 1. Tipos de Microplásticos

El sistema clasifica tres categorías principales:

**FRA (Fragmentos):**
- Piezas irregulares de plástico fragmentado
- Forma irregular, baja circularidad
- Brillo medio a alto

**PEL (Pellets):**
- Gránulos esféricos o cilíndricos de plástico
- Alta circularidad (≥0.78)
- Forma compacta y regular

**TAR (Alquitrán):**
- Partículas de alquitrán (no microplástico)
- Circularidad variable
- Brillo bajo característico (≤182.5)

---

### 2. Augmentación de Datos (Data Augmentation)

Para generar un dataset robusto a partir de imágenes base limitadas en caso de implementar un algoritmo de clasificacion ej: Random Forest:

**Transformaciones Aplicadas:**
- **Espejados**: Horizontal y vertical
- **Rotaciones**: 90°, 180°, 270°
- **Modificación de Brillo**: Factores 0.8 (oscurecer) y 1.2 (aclarar)
- **Ruido Gaussiano**: σ = 20
- **Desenfoque**: Kernel 5x5

**Resultado:**
- De 2 imágenes base por clase → ~20 variantes por clase
- Total: ~60 imágenes aumentadas para entrenamiento
- Mayor robustez ante variaciones de iluminación y orientación

---

### 3. Extracción de Características

El sistema calcula dos características principales por región:

**Brillo Promedio (Canal V en HSV):**
```python
brillo_promedio = np.mean(img_hsv[..., 2])
```
- Mide la luminosidad general de la partícula
- Rango: [0, 255]
- Útil para distinguir alquitrán (oscuro) de plástico (más claro)

**Circularidad Ponderada por Área:**
```python
circularidad = 4 * π * área / perímetro²
```
- Mide qué tan circular es la forma (1.0 = círculo perfecto)
- Ponderada por área de cada contorno detectado
- Discrimina pellets (alta circularidad) de fragmentos (baja circularidad)

**Preprocesamiento:**
1. Conversión a escala de grises
2. Umbralización adaptativa (método Otsu)
3. Operaciones morfológicas (cierre) para limpiar ruido
4. Detección de contornos externos
5. Filtrado por área mínima (>100 píxeles)

---

### 4. Clasificador Basado en Reglas

Sistema basado en reglas a partir de las caracteristicas extraidas:

```python
def clasificar(img):
    brillo, circularidad = extraer_caracteristicas(img)
    
    if circularidad > 0.78:
        return "PEL"
    elif brillo <= 182.5:
        return "TAR"
    else:
        return "FRA"
```

**Umbrales Optimizados:**
- Circularidad > 0.78 → Pellet (forma redondeada)
- Brillo ≤ 182.5 → Alquitrán (oscuro)
- Por defecto → Fragmento

---

### 5. Evaluación del Modelo

**Imagen de Test:**
- `MPs_test.jpg` con anotaciones ground truth
- `MPs_test_bbs.csv` con bounding boxes y etiquetas

**Métricas Calculadas:**
- **Accuracy**: ~69%
- **Matriz de Confusión**: Muestra confusiones entre clases
- **Análisis por Región**: Brillo y circularidad de cada detección

**Visualizaciones:**
- Imagen con predicciones coloreadas por clase
- Matriz de confusión con heatmap
- Tabla de características extraídas

**Código de Colores:**
- 🔴 Rojo: Fragmentos (FRA)
- 🟢 Verde: Pellets (PEL)  
- 🔵 Azul: Alquitrán (TAR)

---

### 6. Limitaciones y Mejoras Futuras

**Limitaciones Actuales:**
- Clasificador simple basado solo en 2 características
- Sensible a condiciones de iluminación extremas
- No considera información de color

**Mejoras Propuestas:**
- Usar más características geométricas (según [SMACC paper](https://doi.org/10.1109/ACCESS.2020.2970498)):
  - Relación de aspecto del contenedor
  - Relación entre ejes de elipse ajustada
  - Relación de distancias centroide-contorno
- Implementar clasificador de machine learning (SVM, Random Forest)
- Aumentar dataset con más imágenes reales
- Considerar características de textura (GLCM, LBP)
- Implementar segmentación más avanzada (watershed, grabcut)

---

## 🎮 Uso

### Identificación de Monedas

### Ejecución Principal

```bash
python coin_detector.py
```

### Flujo de Trabajo

1. **Carga de Imagen**: El sistema carga automáticamente `assets/Monedas.jpg`
2. **Detección**: Se detectan las monedas usando el método configurado
3. **Calibración**: Se abre una ventana interactiva:
   - Haz clic en una moneda de 1€ para calibrar
   - Cierra la ventana cuando hayas terminado
4. **Clasificación**: El sistema clasifica automáticamente todas las monedas
5. **Resultados**: Se muestra la visualización y el resumen detallado

---

### Clasificación de Microplásticos

**Ejecución desde Jupyter Notebook:**

```bash
jupyter notebook vc_p3.ipynb
```

**Celdas del Notebook:**
1. **Importación y configuración**: Carga de paquetes y rutas
2. **Data augmentation**: Generación de variantes
3. **Visualización de dataset**: Vista general de imágenes aumentadas
4. **Extracción de características**: Función `obtener_caracteristicas()`
5. **Entrenamiento/Ajuste**: Análisis de características por clase
6. **Clasificación**: Función `clasificar_por_reglas()`
7. **Evaluación**: Predicción sobre test set con métricas
8. **Visualización de resultados**: Matriz de confusión y predicciones

---

## 🛠️ Requisitos

**Librerías Principales:**
```bash
opencv-python >= 4.5.0
numpy >= 1.19.0
matplotlib >= 3.3.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
seaborn >= 0.11.0
```

## 📁 Estructura del Proyecto

```
vc_p3/
├── coin_detector.py          # Sistema de detección de monedas
├── vc_p3.ipynb              # Notebook principal con ambas tareas
├── README.md                # Esta documentación
└── assets/
    ├── Monedas.jpg          # Imagen de prueba con monedas
    ├── centimos.jpeg        # Imagen adicional de monedas
    └── microplasticos/      # Dataset de microplásticos
        ├── FRA.png          # Muestra de fragmentos
        ├── PEL.png          # Muestra de pellets
        ├── TAR.png          # Muestra de alquitrán
        ├── fragment-03-olympus-10-01-2020.JPG
        ├── pellet-03-olympus-10-01-2020.JPG
        ├── tar-03-olympus-10-01-2020.JPG
        ├── MPs_test.jpg     # Imagen de test
        ├── MPs_test_bbs.csv # Anotaciones ground truth
        └── augmentation/    # Imágenes aumentadas (generadas)
```

---

## 🔬 Detalles Técnicos

### Algoritmo de Ajuste de Círculo (Método de Kasa)

Utiliza mínimos cuadrados algebraicos para ajustar un círculo a los puntos del contorno:

1. **Sistema de ecuaciones**: `[x, y, 1] * [A, B, C] = -(x² + y²)`
2. **Resolución**: Mínimos cuadrados usando `np.linalg.lstsq`
3. **Cálculo del centro**: `cx = -A/2`, `cy = -B/2`
4. **Cálculo del radio**: `r = √(cx² + cy² - C)`
5. **Fallback**: En caso de error, usa `cv2.minEnclosingCircle`

### Refinamiento de Radio

Para mayor robustez ante bordes incompletos o ruidosos:

1. Calcula la mediana de distancias desde el centro a todos los puntos del contorno
2. Compara con el radio del ajuste de Kasa
3. Si la diferencia es < 35%, promedia ambos valores
4. Reduce impacto de outliers y oclusiones parciales

### Preprocesamiento de Imagen

**Pipeline del método avanzado:**
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. Filtro bilateral (preserva bordes, suaviza regiones uniformes)
3. Desenfoque Gaussiano (kernel 5x5)
4. Umbralización adaptativa (blockSize=41, C=10)
5. Cierre morfológico (2 iteraciones, kernel elíptico 7x7)
6. Apertura morfológica (1 iteración)

---

## 📊 Ejemplos de Salida

```
======================================================================
RESUMEN DE DETECCIÓN DE MONEDAS
======================================================================

Valor    Cantidad   Diám.(mm)            Subtotal  
----------------------------------------------------------------------
2.00€    2          25.6, 25.8            4.00€
1.00€    3          23.1, 23.3, 23.2      3.00€
0.50€    1          24.3                  0.50€
0.20€    2          22.1, 22.4            0.40€
----------------------------------------------------------------------
TOTAL    8/8 identificadas               7.90€
======================================================================
```

---

## 💡 Notas de Implementación

### Monedas

- **Iluminación**: Para mejores resultados, usar iluminación uniforme sin sombras fuertes.
- **Oclusiones**: Las monedas parcialmente ocultas pueden detectarse pero clasificarse con baja confianza.
- **Moneda de referencia**: Preferir monedas grandes (1€ o 2€) para mayor precisión en la calibración.
- **Área mínima**: Ajustar `min_area` según la resolución de la imagen (200px² para imágenes ~2000px de ancho).

### Microplásticos

- **Umbrales de clasificación**: Los valores 0.78 (circularidad) y 182.5 (brillo) fueron determinados empíricamente. Pueden requerir ajuste según condiciones de captura.
- **Preprocesamiento**: CLAHE puede mejorar resultados con iluminación no uniforme.
- **Umbral Otsu**: Funciona bien para estas imágenes pero puede fallar con fondos muy heterogéneos.
- **Características adicionales**: El paper SMACC sugiere 7 características geométricas; este proyecto usa 2 para simplicidad.
- **Validación cruzada**: Para evaluación más robusta, considerar k-fold cross-validation.

---

## 📚 Referencias

- **SMACC Paper**: [A System for Microplastics Automatic Counting and Classification](https://doi.org/10.1109/ACCESS.2020.2970498)
- **OpenCV Documentation**: [Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- **Método Kasa**: Algebraic circle fitting for improved robustness in contour analysis
- **Data Augmentation**: Técnicas estándar de aumento de datos en visión por computador

---

## 📝 Autor

Giancarlo Prado Abreu
Proyecto desarrollado para la asignatura de Visión por Computador