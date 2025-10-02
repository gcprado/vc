# VC_P2 - Visión por Computador: Análisis, Procesamiento de Imágenes y Seguimiento de Objetos

Proyecto 2 de Visión por Computador que implementa técnicas avanzadas de análisis y procesamiento de imágenes, tanto estáticas como en tiempo real, además de un sistema de seguimiento de objetos por color con efectos visuales.

## 📋 Descripción General

Este notebook implementa cuatro sistemas principales de visión por computador:

1. **Análisis Estático de Imágenes**: Detección de líneas dominantes (filas/columnas) usando Canny y Sobel.
2. **Visualización Sobel en Tiempo Real**: Aplicación interactiva del operador Sobel con múltiples ventanas.
3. **Procesamiento de Imágenes en Tiempo Real**: Múltiples modos de visualización incluyendo Diferencia, Canny, Sobel y Sustracción de Fondo.
4. **Seguimiento de Objetos por Color**: Sistema que genera trazas que se desvanecen temporalmente al detectar determinados objetos.

---

## 🚀 Características Principales

### 1. Análisis Estático de Imágenes con Detección de Líneas

Sistema de análisis que detecta y visualiza líneas horizontales y verticales prominentes en imágenes.

**Algoritmos Implementados:**

**Detector Canny:**
- Aplicación del filtro Canny con umbrales 100 y 200
- Conteo de píxeles blancos por fila y columna
- Umbralización al 90% del máximo para identificar líneas dominantes
- Normalización respecto al total de píxeles por línea

**Detector Sobel con Umbral Otsu:**
- Suavizado Gaussiano (kernel 3x3) previo
- Cálculo de gradientes en X e Y
- Combinación de gradientes (Sobel X + Y)
- Umbralización automática usando método Otsu
- Detección de líneas dominantes con mismo criterio del 90%

**Visualizaciones Incluidas:**

**Vista Separada (2x2):**
- Filas detectadas sobre Canny (azul)
- Columnas detectadas sobre Canny (verde)
- Gráfico de distribución de píxeles por fila con umbral
- Gráfico de distribución de píxeles por columna con umbral

**Vista Combinada (1x2):**
- Filas y columnas juntas sobre Canny
- Filas y columnas sobre imagen RGB original

**Estadísticas Calculadas:**
- Número de filas detectadas
- Número de columnas detectadas
- Promedio de % píxeles blancos por fila/columna
- Localización exacta (índices) de filas y columnas detectadas
- Comparación entre detecciones Sobel vs Canny

---

### 2. Visualización Sobel en Tiempo Real

Demostración interactiva del operador Sobel con 4 ventanas simultáneas:

- Original: Feed de cámara con efecto espejo
- Sobel X: Detección de bordes verticales
- Sobel Y: Detección de bordes horizontales
- Sobel X+Y: Combinación de ambos gradientes

**Características Técnicas:**
- Suavizado Gaussiano (3x3) para reducir ruido
- Conversión a 8 bits para visualización correcta
- Actualización en tiempo real (~30 FPS)

---

### 3. Procesamiento de Imágenes en Tiempo Real

El sistema ofrece 5 modos diferentes de procesamiento:

- **Original**: Visualización directa del feed de la cámara con efecto espejo
- **Diferencia**: Detección de movimiento mediante diferencia absoluta entre frames consecutivos
- **Sobel**: Detección de bordes usando el operador Sobel con identificación de líneas horizontales y verticales prominentes
- **Canny**: Detección de bordes mediante el algoritmo Canny con análisis de líneas principales
- **Fondo**: Sustracción de fondo usando algoritmo MOG2 para segmentación de objetos en movimiento

**Características Técnicas:**
- Umbralización adaptativa (OTSU)
- Sustracción de fondo con detección de sombras

---

### 4. Seguimiento de Objetos por Color con Trazas Desvanecientes

Sistema de seguimiento multi-color con efectos visuales temporales.

**Colores Soportados:**
- **Rojo** 🔴
- **Verde** 🟢
- **Azul** 🔵
- **Amarillo** 🟡

**Modos de Seguimiento:**
- **Modo Individual**: Seguimiento de un solo color a la vez
- **Modo Multicolor**: Seguimiento simultáneo de los 4 colores

**Efectos Visuales:**
- Trazas que se desvanecen automáticamente después de 2.5 segundos
- Grosor dinámico de la traza (8px → 1px) según la antigüedad
- Transparencia progresiva (100% → 20%) para efecto de desvanecimiento suave
- Indicador visual del objeto detectado con círculo y punto central

**Procesamiento:**
- Rangos HSV optimizados para cada color
- Filtrado Gaussiano para reducción de ruido
- Operaciones morfológicas (erosión, dilatación, cierre)
- Filtrado por área mínima (500 píxeles) para eliminar falsos positivos
- Seguimiento basado en contornos más grandes

---

## 🎮 Controles

**Procesamiento de Imágenes:**
- `m`: Cambiar entre modos de procesamiento
- `ESC`: Salir del programa

**Seguimiento de Colores:**
- `m`: Cambiar color a seguir (modo individual)
- `a`: Alternar entre modo individual y multicolor
- `ESC`: Salir del programa

---

## 🛠️ Requisitos

```bash
opencv-python >= 4.5.0
numpy >= 1.19.0
