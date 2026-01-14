<p align="center">
  <img src="assets/portada.png" alt="Portada del proyecto" width="1024"/>
</p>

## Motivación

La gestión de inventarios en **entornos comerciales y logísticos** modernos representa un desafío cada vez más complejo debido al **alto volumen y diversidad de productos**, la necesidad de mantener información actualizada en **tiempo real** y la reducción de errores humanos en los procesos de control de stock. Tradicionalmente, estas tareas se han realizado de **forma manual** o semiautomática, lo que conlleva costes elevados, falta de escalabilidad y una alta probabilidad de inconsistencias en los datos.

En los últimos años, los avances en **visión por computador** y **aprendizaje profundo** han demostrado un gran potencial para **automatizar tareas** de percepción visual como la detección, segmentación y reconocimiento de objetos. **Modelos del estado del arte** como permiten abordar problemas que hasta hace poco requerían **intervención humana directa**.

Este trabajo surge de la motivación de explorar y **aplicar estas tecnologías** en un problema realista y de alto impacto práctico: la **automatización del conteo y la gestión de inventario mediante sistemas de visión artificial**. Además, se busca no solo implementar una solución funcional, sino también analizar, comparar y comprender las fortalezas y limitaciones de los distintos enfoques existentes, evaluando su viabilidad en escenarios reales de uso.

## Objetivos del Trabajo

### Objetivo General

Desarrollar y evaluar un sistema basado en visión por computador **capaz de contar y clasificar productos en estanterías**, con el fin de automatizar tareas de control y gestión de inventario en entornos comerciales o logísticos.

### Objetivos Específicos

- Estudiar y aplicar modelos del estado del arte en detección y segmentación de objetos, como **YOLO y SAM**.  
- Entrenar y evaluar un modelo de detección sobre un dataset realista de alta densidad de objetos (**SKU110K**).  
- Integrar modelos multimodales como **CLIP** para el reconocimiento de productos mediante clasificación zero-shot.  
- Aplicar métodos de **clustering** como **K-Means y DBSCAN** sobre **embeddings de productos** reducidos con **PCA** para agruparlos automáticamente según su similitud visual.
- Diseñar un pipeline completo que permita:
  - Detectar productos en imágenes y vídeo.  
  - Clasificar productos y contar instancias
  - Realizar un seguimiento temporal para generar gráficos de evolución de inventario por producto y realizar predicciones sobre stock.
- Analizar el rendimiento de los distintos enfoques mediante métricas cuantitativas y comparativas experimentales.  
- Evaluar la viabilidad del uso de estos sistemas como apoyo a la gestión automática de inventario en escenarios reales.


## Descripción técnica del trabajo realizado

El pipeline propuesto para el sistema de gestión de inventario es el siguiente:

**Imágenes → Detección → Clasificación → Persistencia de resultados → Visualización de resultados**

De este modo, el primer paso del pipeline consiste en realizar la detección de los productos presentes en las imágenes. Para ello, se propone entrenar un modelo de detección de objetos en tiempo real basado en YOLOv8 de Ultralytics, adaptado al dataset específico de estanterías comerciales.

### Dataset y Entrenamiento del Modelo

#### Dataset SKU110K
El dataset **SKU110K** es un conjunto de imágenes de **estanterías comerciales** con **alta densidad de productos**, desarrollado para tareas de detección y conteo de objetos en entornos de retail. Contiene más de **11,000 imágenes** y alrededor de **400,000 instancias de productos**, etiquetadas mediante bounding boxes. Las imágenes incluyen múltiples desafíos típicos de inventario real, como superposición de productos, variaciones de iluminación, perspectivas diferentes, y oclusiones parciales, lo que lo convierte en un dataset adecuado para evaluar modelos de detección y segmentación en entornos logísticos.

El dataset SKU110K se organizó siguiendo la estructura estándar de YOLO:

```
.
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```

**Training set:** Este subconjunto contiene 8,219 imágenes y sus anotaciones, utilizadas para entrenar los modelos de detección de objetos.  

**Validation set:** Este subconjunto consta de 588 imágenes y anotaciones, usadas para la validación del modelo durante el entrenamiento.  

**Test set:** Este subconjunto incluye 2,936 imágenes, diseñadas para la evaluación final de los modelos entrenados de detección de objetos.

**Archivo YAML oficial para el entrenamiento con Ultralytics:** [Ver aquí](https://docs.ultralytics.com/datasets/detect/sku-110k/#dataset-yaml)  
**Artículo original del dataset SKU110K:** [**Goldman, E., Herzig, R., Eisenschtat, A., Goldberger, J., & Hassner, T. (2019).** *Precise Detection in Densely Packed Scenes.* In *Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR).*](https://arxiv.org/pdf/1904.00853)


#### Modelo y entrenamiento
Se utilizó un modelo base **YOLOv8n** preentrenado en COCO, adaptado al dataset.  El entrenamiento se realizó durante 10 épocas con imágenes de 640×640 píxeles y batch size de 8. 

El modelo resultante se guarda en: `yolov8_train_results/content/runs/detect/train/weights/best.pt`

---

### Resultados del Entrenamiento

Durante el proceso de entrenamiento se generaron diversas gráficas que permiten analizar el rendimiento del modelo.

A continuación se incluyen los enlaces a las principales métricas visuales:

| Métrica | Imagen |
|----------|--------|
| F1-Score vs Confidence | ![BoxF1_curve](yolov8_train_results/content/runs/detect/train/BoxF1_curve.png) |
| Precision vs Confidence | ![BoxP_curve](yolov8_train_results/content/runs/detect/train/BoxP_curve.png) |
| Precision vs Recall | ![BoxPR_curve](yolov8_train_results/content/runs/detect/train/BoxPR_curve.png) |
| Recall vs Confidence | ![BoxR_curve](yolov8_train_results/content/runs/detect/train/BoxR_curve.png) |
| Confusion Matrix | ![confusion_matrix](yolov8_train_results/content/runs/detect/train/confusion_matrix.png) |
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

La validación se realizó sobre el conjunto de prueba (split *test*), utilizando el modelo **YOLOv8n** entrenado.

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

Una vez entrenado el modelo **YOLOv8**, este es capaz de detectar los productos presentes en la escena con una **precisión del 88.3%**.  

A continuación se muestran ejemplos de detecciones en diferentes secciones de un comercio, ilustrando el rendimiento del modelo:

![Detecciones-Yolo-1](assets/resultados_deteccion_1.png)
![Detecciones-Yolo-2](assets/resultados_deteccion_2.png)
![Detecciones-Yolo-3](assets/resultados_deteccion_3.png)
![Detecciones-Yolo-4](assets/resultados_deteccion_4.png)

En un principio, conocer la cantidad total de productos es relativamente sencillo: basta con contar la cantidad de **bounding boxes** detectadas en la imagen.

| SECCIÓN               | PRODUCTOS |
|----------------------|-----------|
| SECCION-ALIMENTOS-1  | 141       |
| SECCION-ALIMENTOS-2  | 164       |
| SECCION-ASEO-1       | 130       |
| SECCION-ASEO-2       | 123       |
| SECCION-BEBIDAS-1    | 179       |
| SECCION-BEBIDAS-2    | 134       |
| SECCION-BEBIDAS-3    | 148       |
| SECCION-LIMPIEZA-1   | 252       |
| SECCION-LIMPIEZA-2   | 252       |
| SECCION-LIMPIEZA-3   | 236       |
| SECCION-MEDICAMENTOS-1 | 144     |

#### Rendimiento del modelo en situaciones no ideales

Iluminacion no uniforme:
![alt text](assets/iluminacion.jpg)

Alto contraste:
![alt text](assets/contraste.jpg)

Motion blur:
![alt text](assets/motion-blur.jpg)

### Utilizacion de SAM para la realizacion de deteccionoes

Inicialmente se consideró el uso de **modelos de segmentación**, en particular **SAM** (Segment Anything Model), como alternativa para la detección de productos en estanterías. Sin embargo, este enfoque resultó poco adecuado para el escenario planteado.

![alt text](image.png)
![alt text](image-1.png)

En entornos **densamente poblados**, como los estantes de retail, los modelos de segmentación tienden a presentar dificultades importantes: Estos modelos funcionan mejor en escenas donde los objetos a segmentar son **semánticamente distintos** del fondo, mientras que en estanterías comerciales existe una gran cantidad de productos **visualmente muy similares entre sí**, lo que dificulta separar correctamente **cada instancia del fondo** y de los **objetos vecinos**.  

Por último, el **entrenamiento** o ajuste fino de modelos de segmentación requiere **anotaciones en forma de máscaras**, que son mucho menos comunes y más costosas de producir que las anotaciones mediante *bounding boxes*. Esto limita la disponibilidad de conjuntos de datos adecuados y hace que este enfoque sea menos práctico en comparación con métodos basados en detección por cajas delimitadoras.

---

### Procesamiento de Imagenes

El siguiente desafío consiste en **clasificar cada uno de los productos detectados**. Para ello, se plantean dos enfoques:

1. **Clasificación directa del ROI con zero-shot por un modelo multimodal (CLIP):**  
   Este enfoque permite **asignar etiquetas** a los productos sin necesidad de un **entrenamiento específico** para cada categoría, aprovechando la capacidad del modelo de relacionar imágenes y texto de manera directa.

2. **Extraccion de features mediante embeddings generados por modelos self-supervised (DINOv2):**  
   Este método consiste en representar cada producto mediante un **embedding visual** y comparar estas representaciones entre sí, permitiendo agrupar o identificar productos similares basándose en su similitud en el espacio de características aprendido.

#### OpenCLIP (LAION-2B)

El dataset **LAION-5B** es un conjunto masivo de **5,85 mil millones de pares imagen-texto**, desarrollado para investigación en modelos multimodales a gran escala. Representa un incremento de más de 14 veces respecto a su predecesor, **LAION-400M**, anteriormente el dataset abierto más grande del mundo. Aproximadamente 2,3 mil millones de muestras están en inglés, 2,2 mil millones en más de 100 idiomas adicionales, y 1 mil millón contiene textos sin asignación lingüística clara (por ejemplo, nombres propios).  

LAION-5B ha sido diseñado para permitir la investigación de modelos de imagen-texto a gran escala, y ha servido como base para entrenar modelos tipo CLIP de manera reproducible y accesible públicamente.

**Artículo original sobre CLIP y LAION-5B:** [**Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G. et al. (2021).** *Learning Transferable Visual Models From Natural Language Supervision.* In *International Conference on Machine Learning (ICML).*](https://laion.ai/blog/laion-5b/)

Para la identificación de productos, se utilizaron los modelos **CLIP ViT-B/32 y CLIP ViT-L/14 entrenado con LAION-2B**, un **subconjunto filtrado de LAION-5B** que contiene pares imagen-texto seleccionados mediante CLIP para un tamaño más manejable y para facilitar el entrenamiento reproducible de modelos zero-shot.  

Estos modelos genera embeddings de imagen y texto que permiten realizar clasificación **zero-shot** de ROIs detectados por YOLO. 

La figura muestra la disposición de los productos en la sección de un comercio seleccionada para el estudio.

![imagen-base](assets/seccion-bebidas.jpg)

Se utilizó YOLO para identificar las regiones correspondientes a cada producto. La imagen siguiente muestra las detecciones obtenidas:

![detecciones](assets/seccion-bebidas-deteccion.jpg)

Cada región de interés identificada se envió al modelo CLIP para su clasificación automática en la correspondiente categoría de producto.

=== Clase: coke diet (30 imágenes) ===
![Clasificacion directa 1](assets/directo-1.png)

=== Clase: dr pepper diet (16 imágenes) ===
![Clasificacion directa 2](assets/directo-2.png)

=== Clase: dr pepper (20 imágenes) ===
![Clasificacion directa 3](assets/directo-3.png)

=== Clase: coca cola bottle (12 imágenes) ===
![Clasificacion directa 4](assets/directo-4.png)

#### DINOv2

DINOv2 es un modelo **self-supervised** para aprendizaje de representaciones visuales, capaz de **generar embeddings** de alta calidad que permitirá **comparar productos y realizar análisis mediante clustering** de los objetos detectados.

**Artículo original sobre DINOv2:** [**Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M. et al. (2023).** *DINOv2: Learning Robust Visual Features without Supervision.* In Transactions on Machine Learning Research (TMLR).*](https://arxiv.org/pdf/2304.07193)

Para la generación de embeddings de los productos, se utilizó el modelo **DINOv2 ViT-L/14**

En este trabajo, los embeddings generados por DINOv2 se utilizan para representar cada producto detectado por YOLO, permitiendo aplicar técnicas de **clustering y comparación de similitud** entre productos, lo que facilita la agrupación automática y el análisis de inventario visual.

A continuación, se pone a prueba este enfoque en el mismo escenario descrito previamente, partiendo de las mismas detecciones iniciales obtenidas con YOLO:

![Resultados de detección](assets/seccion-bebidas-deteccion.jpg)

Se procede a generar los embeddings de cada región de interés, los cuales se agrupan mediante **HDBSCAN** y se visualizan utilizando **PCA** y **UMAP** para explorar las agrupaciones obtenidas.

![Clusters](assets/clusters.png)

Luego de que se hayan generado estos embedings se procede a clusterizar:

![Cluster 0](assets/cluster-0.png)
![cluster 1](assets/cluster-1.png)
![cluster 2](assets/cluster-2.png)
![cluster 3](assets/cluster-3.png)
![cluster 4](assets/cluster-4.png)
![cluster 5](assets/cluster-5.png)
![cluster 6](assets/cluster-6.png)
![cluster 7](assets/cluster-7.png)
![cluster 8](assets/cluster-8.png)
![cluster 9](assets/cluster-9.png)
![cluster 10](assets/cluster-10.png)
![cluster 11](assets/cluster-11.png)

A diferencia del enfoque anterior, en el que se procesaba cada ROI individual y se asignaban las etiquetas directamente. En este enfoque se extraen los **ROI's mas representativos de cada cluster**, esto se realiza sacando dentro de un cluster los **embeddings mas cercanos al centroide del mismo**, se procede a clasificarlos y elegir la categoria mediante un votación ponderada conjunta.

Este método permite asignar etiquetas de forma **más confiable**, ya que se consideran los **elementos centrales** del cluster y se evita que **ROIs difíciles de clasificar** por CLIP reciban etiquetas incorrectas.

Esto tiene varias ventajas:

1. **Ahorro de recursos:** solo es necesario procesar los elementos representativos del cluster, en lugar de cada ROI individual.

2. **Mayor robustez:** al usar una votación múltiple, los ROIs más difíciles de clasificar (por ejemplo, los con embeddings ambiguos) tienen más probabilidades de recibir la etiqueta correcta.

Ahora, para la votación que determina la clasificación del producto, se exploraron **dos alternativas** posibles. Por un lado, se puede realizar la clasificación directamente con **CLIP nuevamente**. O por otro lado, se podría optar por un **matching basado en embeddings**, es decir, encontrar el producto más cercano mediante un **nearest neighbor** en el **espacio de características**.

A continuación, se procede a extraer los **ROI más representativos** de cada cluster generado. En este ejemplo se seleccionan tres ROI por cluster como demostración, aunque el enfoque permite utilizar **n ROI**, asignando a cada uno un **peso normalizado** que refleje su importancia relativa dentro del cluster.

![ROI 0](assets/roi-0.png)
![ROI 1](assets/roi-1.png)
![ROI 2](assets/roi-2.png)

Este mismo procedimiento se aplica al resto de clusters identificados en la sección.

---

Cada ROI seleccionado se clasifica mediante el modelo **CLIP**. Posteriormente, se realiza una **votación ponderada** considerando los pesos asignados a cada ROI, y se asigna el **predicted label** correspondiente a todos los elementos que pertenecen al mismo cluster en la base de datos.

![Predicción 1](assets/prediccion-1.png)
![Predicción 2](assets/prediccion-2.png)
![Predicción 3](assets/prediccion-3.png)
![Predicción 4](assets/prediccion-4.png)
![Predicción 5](assets/prediccion-5.png)
![Predicción 6](assets/prediccion-6.png)

---

### Resultados Obtenidos

Se presenta la tabla de resultados obtenidos al aplicar la **clasificación directa** desde YOLO hasta LVM, mostrando el número de predicciones por clase, cuántas coincidieron con la etiqueta real y la precisión por clase. Esta tabla permite evaluar el efecto acumulado de los errores en las 3 etapas:

| Product              | Predicted Count| Predicted Correct | Real Count | Accuracy (%) |
|----------------------|----------------|-----------------|------------|--------------|
| coke diet            | 30             | 28               | 30         | 93.33%      |
| coca cola red        | 84             | 84               | 93         | 90.32%       |
| sprite               | 34             | 32               | 33         | 96.96%       |
| dr pepper            | 20             | 10               | 10         | 50.00%       |
| dr pepper diet       | 16             | 13               | 16         | 81.25%      |
| dr pepper 23         | 21             | 21               | 30         | 70.00%       |
| fanta                | 4              | 4                | 4          | 100.00%      |
| coca cola black      | 16             | 14               | 16         | 87.50%      |
| coca cola bottle     | 12             | 4                | 4          | 33.33%       |

La **precisión media** obtenida para este pipeline directo es de **78.08%**.

---

A continuación, se muestra una tabla comparativa que refleja los resultados acumulados considerando todos los pasos del pipeline: detecciones iniciales, clusterización de ROI y clasificación mediante CLIP. Esta tabla permite evaluar el efecto acumulado de los **errores en las 3 etapas**:

| Product              | Predicted Count | Predicted Correct | Real Count | Accuracy (%) |
|----------------------|-----------------|-------------------|------------|--------------|
| coke diet            | 29              | 29                | 30         | 96.67%       |
| coca cola red        | 88              | 88                | 93         | 94.62%       |
| sprite               | 27              | 27                | 33         | 81.82%       |
| dr pepper            | 9               | 8                 | 10         | 80.00%       |
| dr pepper diet       | 16              | 16                | 16         | 100.00%      |
| dr pepper 23         | 24              | 24                | 30         | 80.00%       |
| fanta                | 4               | 4                 | 4          | 100.00%      |
| coca cola black      | 12              | 12                | 16         | 75.00%       |
| coca cola bottle     | 4               | 4                 | 4          | 100.00%      |


La **precisión media** para el pipeline completo se incrementa a **89.79%**, reflejando la mejora obtenida al integrar la clusterización y la votación ponderada en la clasificación de los ROI.

No se observaron diferencias significativas entre **ViT-B/32 y ViT-L/14**, lo cual sugiere que en este problema la complejidad del modelo encargado de zero-shot no es el factor limitante. Las principales fuentes de error provienen de la calidad de los ROIs, la similitud visual entre productos y las limitaciones en las detecciónes y clustering, por lo que un modelo base resulta suficiente.

![ruido](assets/ruido.png)

---

#### Generacion de dashboards visuales

A partir de los datos obtenidos, es posible generar **gráficos visuales** que permiten observar la evolución del stock de los productos a lo largo del tiempo, así como estimar cuándo se agotarán.

Por ejemplo, para esta sección de un comercio se registraron varias imágenes que muestran la evolución del inventario y la disposición de los productos en distintos momentos:

![momento-1](assets/momento-1.png)

| fiesta stawberry | fiesta lime | fiesta cola | fiesta root beer | fiesta orange | fiesta grape | island sun peaches | timestamp |
|-----------------|------------|------------|-----------------|---------------|--------------|------------------|----------------------------|
| 20              | 27         | 24         | 27              | 23            | 24           | 79               | 2026-01-13 05:24:38.359674 |
| 17              | 23         | 20         | 22              | 19            | 19           | 71               | 2026-01-13 11:24:38.359674 |
| 15              | 19         | 18         | 18              | 16            | 16           | 62               | 2026-01-13 17:24:38.359674 |
| 12              | 15         | 15         | 15              | 14            | 14           | 54               | 2026-01-13 23:24:38.359674 |
| 10              | 13         | 13         | 13              | 11            | 11           | 49               | 2026-01-14 05:24:38.359674 |

La siguiente figura muestra la comparación del stock de productos en **dos instantes diferentes**:

![Comparación entre dos momentos](assets/grafica-comparacion.png)

---

Se calcula el **cambio neto** de cada producto entre los dos momentos, destacando cuáles han disminuido más rápidamente:

![Cambio neto](assets/grafica-cambio.png)

---

El siguiente gráfico representa la **evolución del inventario a lo largo del tiempo**, mostrando cómo varían las cantidades disponibles en cada snapshot registrado:

![Evolución temporal](assets/grafica-tiempo.png)

---

A partir de la tendencia de disminución, se generan predicciones sobre **cuándo ciertos productos podrían agotarse**:

![Predicción 1](assets/grafica-prediccion-1.png)
![Predicción 2](assets/grafica-prediccion-2.png)
![Predicción 3](assets/grafica-prediccion-3.png)

---

## Conclusiones finales

En este proyecto se desarrolló un pipeline completo para la clasificación automática de productos a partir de ROIs, combinando clustering por embeddings y clasificación semántica mediante CLIP. El uso de ROIs representativos y votación ponderada por cluster permitió reducir el ruido y mejorar la estabilidad de las predicciones frente a la clasificación individual. Los resultados muestran una alta precisión en la mayoría de las categorías, aunque persisten errores en clases visualmente similares. Finalmente, se exploraron alternativas basadas en matching por embeddings, que abren la puerta a sistemas más eficientes y escalables sin necesidad de reclasificar con modelos pesados en cada iteración.

Asimismo, aunque se exploró el uso de modelos de segmentación como SAM, se concluyó que este tipo de enfoques no resulta práctico en estanterías comerciales debido a la alta similitud visual entre productos, la complejidad del posprocesamiento de máscaras y la escasa disponibilidad de datasets con anotaciones de segmentación.

En conjunto, los resultados muestran que el uso de embeddings y estrategias de agregación por cluster constituye una solución eficaz y escalable para este tipo de escenarios.

---

### Propuestas de ampliación

- Extracción de etiquetas para zero-shot mediante módulo de OCR o LLM
- Comparación directa con base de datos de embeddings y FAISS para busqueda
- Desarrollo de una suite de tests automatizados que permita validar el funcionamiento del sistema de forma sistemática y reproducible.  
- Implementación de una aplicación full-stack que integre todo el pipeline y proporcione una interfaz de usuario para facilitar su uso y análisis de resultados.  
- Optimización de los algoritmos implementados para mejorar el rendimiento computacional y reducir los tiempos de procesamiento.  
- Refactorización, modularización y mejora general de la calidad del código para facilitar su mantenimiento y escalabilidad futura.  

---

## Fuentes y tecnologías utilizadas

- **YOLO**: para detección de objetos en imágenes y vídeo, adaptado al dataset SKU110K.  
- **SAM (Segment Anything Model)**: para segmentación automática de productos en estanterías.  
- **OpenCLIP (ViT-B/32 y ViT-L/14)**: para reconocimiento y clasificación zero-shot de productos mediante embeddings multimodales.  
- **DINOv2 y DINOv3**: para extracción de características visuales y análisis de similitudes entre productos.  
- **DBSCAN, PCA y UMAP**: para agrupamiento automático de productos según sus embeddings visuales.  
- **PyTorch, OpenCV, scikit-learn y Ultralytics**: frameworks y librerías para entrenamiento, procesamiento de imágenes y visualización de resultados.

---

## Indicación de herramientas/tecnologías con las que se hubiera gustado contar

- **GPU de mayor capacidad** para acelerar entrenamientos de modelos grandes y experimentos con múltiples configuraciones.  
- **Datasets de retail con máscaras de segmentación**, en lugar de solo bounding boxes, que permitieran entrenar y evaluar modelos de segmentación más precisos.  

---

## Repositorio y codigo fuente:

https://github.com/gcprado/vc/edit/main/IMS
