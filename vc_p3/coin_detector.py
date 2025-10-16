"""
Sistema de Detección y Clasificación de Monedas
Implementación de la tarea de identificación de monedas y cálculo de valor total
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Dimensiones de las monedas de euro en milímetros
COIN_DIMENSIONS = {
    2.00: 25.75,
    1.00: 23.25,
    0.50: 24.25,
    0.20: 22.25,
    0.10: 19.75,
    0.05: 21.25,
    0.02: 18.75,
    0.01: 16.25
}

class CoinDetector:
    """Detector de monedas con calibración interactiva"""
    
    def __init__(self, image_path):
        """
        Inicializa el detector con una imagen
        
        Args:
            image_path: Ruta a la imagen con monedas
        """
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.coins = []  # Lista de monedas detectadas
        self.reference_coin = None  # Moneda de referencia seleccionada
        self.pixels_per_mm = None  # Factor de conversión píxeles a mm
        self.reference_value = None  # Valor de la moneda de referencia
    
    def _fit_circle_kasa(self, contour):
        """
        Ajuste de círculo usando método de Kasa (mínimos cuadrados algebraicos)
        Más preciso que minEnclosingCircle para contornos con ruido
        
        Args:
            contour: Contorno de OpenCV
            
        Returns:
            (cx, cy, radius): Centro y radio del círculo ajustado
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        
        if pts.shape[0] < 3:
            # Fallback a minEnclosingCircle
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            return float(cx), float(cy), float(r)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Sistema de ecuaciones: [x, y, 1] * [A, B, C] = -(x² + y²)
        A_matrix = np.column_stack([x, y, np.ones_like(x)])
        b_vector = -(x*x + y*y)
        
        try:
            solution, *_ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback si falla
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            return float(cx), float(cy), float(r)
        
        A, B, C = solution
        cx = -A / 2.0
        cy = -B / 2.0
        rad_squared = cx*cx + cy*cy - C
        
        if rad_squared <= 0 or not np.isfinite(rad_squared):
            # Fallback
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            return float(cx), float(cy), float(r)
        
        r = np.sqrt(rad_squared)
        return float(cx), float(cy), float(r)
    
    def _median_radius(self, cx, cy, contour):
        """
        Calcula la mediana de distancias desde el centro al contorno
        Más robusto que el promedio ante bordes incompletos
        
        Args:
            cx, cy: Coordenadas del centro
            contour: Contorno de OpenCV
            
        Returns:
            Radio mediano o None si falla
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        distances = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        
        if distances.size == 0:
            return None
        
        return float(np.median(distances))
        
    def detect_coins(self, method='contours_advanced', **params):
        """
        Detecta monedas en la imagen
        
        Args:
            method: 'hough', 'contours', o 'contours_advanced' (recomendado)
            **params: Parámetros adicionales para el método de detección
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        if method == 'contours_advanced':
            # Método avanzado con mejor preprocesado (recomendado)
            # CLAHE para uniformar iluminación
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_cl = clahe.apply(gray)
            
            # Filtro bilateral: reduce ruido manteniendo bordes
            bilateral = cv2.bilateralFilter(gray_cl, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Desenfoque gaussiano suave
            blurred = cv2.GaussianBlur(bilateral, (5,5), 0)
            
            # Threshold adaptativo
            thresh = cv2.adaptiveThreshold(gray_cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 41, 10)
            
            # Morfología: cerrar huecos y limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            min_area = params.get('min_area', 200)
            min_circularity = params.get('min_circularity', 0.55)
            min_solidity = params.get('min_solidity', 0.6)
            
            self.coins = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                
                # Circularidad: 4π·área / perímetro²
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                
                # Solidez: área / área del hull convexo
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull) if len(hull) > 2 else area
                solidity = area / hull_area if hull_area > 0 else 0.0
                
                if circularity < min_circularity or solidity < min_solidity:
                    continue
                
                # Ajuste de círculo con Kasa (más preciso)
                cx, cy, r = self._fit_circle_kasa(contour)
                
                # Refinar radio usando mediana de distancias
                median_r = self._median_radius(cx, cy, contour)
                if median_r and abs(median_r - r) / (r + 1e-6) < 0.35:
                    r = 0.5 * (r + median_r)  # Promedio para robustez
                
                self.coins.append({
                    'x': int(round(cx)),
                    'y': int(round(cy)),
                    'radius': int(round(r)),
                    'diameter_px': int(round(2 * r)),
                    'circularity': circularity,
                    'solidity': solidity
                })
        
        elif method == 'hough':
            # Suavizar imagen para reducir ruido
            blurred = cv2.medianBlur(gray, params.get('blur', 7))
            
            # Detectar círculos con Hough Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params.get('dp', 1),
                minDist=params.get('minDist', 100),
                param1=params.get('param1', 100),
                param2=params.get('param2', 50),
                minRadius=params.get('minRadius', 30),
                maxRadius=params.get('maxRadius', 200)
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                self.coins = [
                    {'x': x, 'y': y, 'radius': r, 'diameter_px': 2*r}
                    for x, y, r in circles
                ]
        
        elif method == 'contours':
            # Umbralizar imagen
            _, binary = cv2.threshold(gray, params.get('threshold', 200), 
                                     255, cv2.THRESH_BINARY_INV)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            self.coins = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > params.get('min_area', 1000):
                    # Calcular círculo mínimo que contiene el contorno
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Calcular circularidad para filtrar no-monedas
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    if circularity > params.get('min_circularity', 0.7):
                        self.coins.append({
                            'x': int(x),
                            'y': int(y),
                            'radius': int(radius),
                            'diameter_px': int(2 * radius),
                            'circularity': circularity
                        })
        
        return len(self.coins)
    
    def interactive_calibration(self, reference_value=1.0):
        """
        Permite seleccionar interactivamente una moneda de referencia
        
        Args:
            reference_value: Valor en euros de la moneda de referencia (ej: 1.0, 0.5, 0.2)
        """
        self.reference_value = reference_value
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.img_rgb)
        ax.set_title(f'Haz clic en una moneda de {reference_value}€ para calibrar')
        ax.axis('off')
        
        # Dibujar monedas detectadas
        for i, coin in enumerate(self.coins):
            circle = plt.Circle((coin['x'], coin['y']), coin['radius'], 
                              color='yellow', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.text(coin['x'], coin['y'], str(i), 
                   color='yellow', ha='center', va='center', fontsize=12, fontweight='bold')
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            # Encontrar la moneda más cercana al clic
            click_x, click_y = event.xdata, event.ydata
            min_dist = float('inf')
            selected_coin = None
            
            for coin in self.coins:
                dist = np.sqrt((coin['x'] - click_x)**2 + (coin['y'] - click_y)**2)
                if dist < min_dist and dist < coin['radius']:
                    min_dist = dist
                    selected_coin = coin
            
            if selected_coin:
                self.reference_coin = selected_coin
                # Calcular factor de conversión
                reference_diameter_mm = COIN_DIMENSIONS[reference_value]
                self.pixels_per_mm = selected_coin['diameter_px'] / reference_diameter_mm
                
                # Resaltar moneda seleccionada
                ax.clear()
                ax.imshow(self.img_rgb)
                ax.set_title(f'Moneda de referencia: {reference_value}€ seleccionada\n' +
                           f'Factor: {self.pixels_per_mm:.2f} px/mm')
                ax.axis('off')
                
                for coin in self.coins:
                    color = 'green' if coin == selected_coin else 'yellow'
                    linewidth = 3 if coin == selected_coin else 2
                    circle = plt.Circle((coin['x'], coin['y']), coin['radius'], 
                                      color=color, fill=False, linewidth=linewidth)
                    ax.add_patch(circle)
                
                plt.draw()
                print(f"✓ Moneda de referencia seleccionada: {reference_value}€")
                print(f"  Diámetro en píxeles: {selected_coin['diameter_px']}")
                print(f"  Diámetro real: {reference_diameter_mm} mm")
                print(f"  Factor conversión: {self.pixels_per_mm:.3f} px/mm")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def classify_coins(self, tolerance_mm=0.8):
        """
        Clasifica las monedas detectadas según su tamaño con rango de tolerancia
        Requiere calibración previa
        
        Args:
            tolerance_mm: Tolerancia en milímetros para clasificación (default 0.8mm)
                         Una moneda se clasifica si está dentro de ±tolerance_mm del valor nominal
        """
        if self.pixels_per_mm is None:
            raise ValueError("Debe calibrar primero con interactive_calibration()")
        
        # Crear rangos de clasificación
        coin_ranges = {
            value: (nominal - tolerance_mm, nominal + tolerance_mm) 
            for value, nominal in COIN_DIMENSIONS.items()
        }
        
        # Clasificar cada moneda
        for coin in self.coins:
            diameter_mm = coin['diameter_px'] / self.pixels_per_mm
            coin['diameter_mm'] = diameter_mm
            
            # Buscar monedas cuyo rango contiene diameter_mm
            matches = [
                value for value, (min_d, max_d) in coin_ranges.items() 
                if min_d <= diameter_mm <= max_d
            ]
            
            if len(matches) == 1:
                # Coincidencia única - clasificación segura
                best_match = matches[0]
                coin['confidence'] = 'high'
            elif len(matches) > 1:
                # Múltiples coincidencias - elegir la más cercana
                best_match = min(matches, key=lambda v: abs(COIN_DIMENSIONS[v] - diameter_mm))
                coin['confidence'] = 'medium'
            else:
                # Ninguna coincidencia en rango - fallback a la más cercana
                best_match = min(COIN_DIMENSIONS.keys(), key=lambda v: abs(COIN_DIMENSIONS[v] - diameter_mm))
                error = abs(COIN_DIMENSIONS[best_match] - diameter_mm)
                if error <= 1.5:  # Aceptar si está dentro de 1.5mm
                    coin['confidence'] = 'low'
                else:
                    coin['confidence'] = 'unknown'
                    best_match = None
            
            coin['value'] = best_match
            if best_match:
                coin['error_mm'] = abs(COIN_DIMENSIONS[best_match] - diameter_mm)
            else:
                coin['error_mm'] = None
    
    def calculate_total(self):
        """Calcula el valor total de las monedas (solo monedas identificadas)"""
        if not self.coins or 'value' not in self.coins[0]:
            raise ValueError("Debe clasificar las monedas primero con classify_coins()")
        
        # Solo sumar monedas identificadas (value no None)
        total = sum(coin['value'] for coin in self.coins if coin['value'] is not None)
        return total
    
    def display_results(self):
        """Muestra los resultados de la detección y clasificación"""
        if not self.coins or 'value' not in self.coins[0]:
            raise ValueError("Debe clasificar las monedas primero con classify_coins()")
        
        # Crear imagen resultado
        img_result = self.img_rgb.copy()
        
        # Contar monedas por valor (solo identificadas)
        coin_counts = {}
        for coin in self.coins:
            value = coin['value']
            if value is not None:
                coin_counts[value] = coin_counts.get(value, 0) + 1
        
        # Colores para cada tipo de moneda
        colors = {
            2.00: (255, 215, 0),    # Dorado
            1.00: (255, 215, 0),    # Dorado
            0.50: (255, 215, 0),    # Dorado
            0.20: (255, 215, 0),    # Dorado
            0.10: (255, 215, 0),    # Dorado
            0.05: (210, 105, 30),   # Cobrizo
            0.02: (210, 105, 30),   # Cobrizo
            0.01: (210, 105, 30)    # Cobrizo
        }
        
        # Dibujar cada moneda con su valor y confianza
        for coin in self.coins:
            # Color según confianza
            confidence = coin.get('confidence', 'unknown')
            if coin['value'] is None or confidence == 'unknown':
                circle_color = (128, 128, 128)  # Gris para desconocidas
                label = "???"
            else:
                # Color según tipo de moneda
                circle_color = colors.get(coin['value'], (0, 255, 0))
                label = f"{coin['value']:.2f}€"
                
                # Ajustar intensidad según confianza
                if confidence == 'low':
                    # Hacer el color más apagado para baja confianza
                    circle_color = tuple(int(c * 0.7) for c in circle_color)
            
            # Grosor según confianza
            thickness = 3 if confidence == 'high' else 2
            
            # Dibujar círculo
            cv2.circle(img_result, (coin['x'], coin['y']), coin['radius'], circle_color, thickness)
            
            # Texto con el valor
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            cv2.rectangle(img_result, 
                        (coin['x'] - text_width//2 - 5, coin['y'] - text_height - 5),
                        (coin['x'] + text_width//2 + 5, coin['y'] + 5),
                        (0, 0, 0), -1)
            
            # Texto
            cv2.putText(img_result, label, 
                       (coin['x'] - text_width//2, coin['y']), 
                       font, font_scale, (255, 255, 255), text_thickness)
        
        # Calcular total
        total = self.calculate_total()
        
        # Crear figura con resultados
        fig = plt.figure(figsize=(16, 8))
        
        # Imagen con monedas etiquetadas
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img_result)
        ax1.set_title(f'Monedas Detectadas y Clasificadas\nTotal: {total:.2f}€', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Tabla resumen
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')
        
        # Preparar datos para la tabla
        table_data = []
        sorted_values = sorted(coin_counts.keys(), reverse=True)
        
        for value in sorted_values:
            count = coin_counts[value]
            subtotal = value * count
            table_data.append([f"{value:.2f}€", count, f"{subtotal:.2f}€"])
        
        table_data.append(['', '', ''])  # Línea separadora
        table_data.append(['TOTAL', len(self.coins), f"{total:.2f}€"])
        
        # Crear tabla
        table = ax2.table(cellText=table_data,
                         colLabels=['Moneda', 'Cantidad', 'Subtotal'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Estilo de la tabla
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Encabezado
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i == len(table_data):  # Total
                    cell.set_facecolor('#FFC107')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax2.set_title('Resumen de Monedas', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir resumen en consola
        print("\n" + "="*70)
        print("RESUMEN DE DETECCIÓN DE MONEDAS")
        print("="*70)
        
        # Mostrar detalles por moneda
        print(f"\n{'Valor':<8} {'Cantidad':<10} {'Diám.(mm)':<20} {'Subtotal':<10}")
        print("-"*70)
        
        for value in sorted_values:
            if value is None:
                continue
            count = coin_counts[value]
            subtotal = value * count
            # Obtener diámetros detectados de esta moneda
            diameters = [f"{c['diameter_mm']:.1f}" for c in self.coins if c['value'] == value]
            diam_str = ', '.join(diameters[:3])  # Mostrar hasta 3
            if len(diameters) > 3:
                diam_str += f" (+{len(diameters)-3})"
            print(f"{value:.2f}€{'':<3} {count:<10} {diam_str:<20} {subtotal:6.2f}€")
        
        # Monedas no identificadas
        unknown_coins = [c for c in self.coins if c['value'] is None]
        if unknown_coins:
            print(f"\n{'???':<8} {len(unknown_coins):<10} {'(no identificadas)':<20} {'---':<10}")
            for c in unknown_coins:
                print(f"  → Diámetro detectado: {c['diameter_mm']:.2f}mm")
        
        print("-"*70)
        identified = len([c for c in self.coins if c['value'] is not None])
        print(f"{'TOTAL':<8} {identified}/{len(self.coins)} identificadas{'':<14} {total:6.2f}€")
        print("="*70 + "\n")


def main():
    """Función principal para ejecutar el detector de monedas"""
    
    # Cargar imagen
    image_path = 'assets/Img4.jpeg'
    
    print("="*60)
    print("SISTEMA DE DETECCIÓN Y VALORACIÓN DE MONEDAS")
    print("="*60)
    
    # Crear detector
    detector = CoinDetector(image_path)
    print(f"\n✓ Imagen cargada: {image_path}")
    print(f"  Dimensiones: {detector.img.shape[1]}x{detector.img.shape[0]} px")
    
    # Detectar monedas con método avanzado
    print("\n[1] Detectando monedas")
    num_coins = detector.detect_coins(
        method='contours',
        min_area=200,
        min_circularity=0.55,
        min_solidity=0.6
    )
    print(f"  ✓ {num_coins} monedas detectadas")
    
    if num_coins == 0:
        print("\n⚠ No se detectaron monedas. Verifica la imagen o ajusta parámetros.")
        return
    
    # Calibración interactiva
    print("\n[2] Calibración interactiva:")
    print("  - Se abrirá una ventana con la imagen")
    print("  - Haz clic en una moneda de 1€ para calibrar")
    print("  - Cierra la ventana cuando hayas terminado")
    detector.interactive_calibration(reference_value=1.0)
    
    if detector.pixels_per_mm is None:
        print("\n✗ Calibración cancelada. No se seleccionó moneda de referencia.")
        return
    
    # Clasificar monedas
    print("\n[3] Clasificando monedas (tolerancia: ±0.8mm)...")
    detector.classify_coins(tolerance_mm=0.8)
    
    # Mostrar resultados de clasificación
    high_conf = sum(1 for c in detector.coins if c.get('confidence') == 'high')
    medium_conf = sum(1 for c in detector.coins if c.get('confidence') == 'medium')
    low_conf = sum(1 for c in detector.coins if c.get('confidence') == 'low')
    unknown = sum(1 for c in detector.coins if c.get('confidence') == 'unknown')
    
    print(f"  ✓ Clasificación completada:")
    print(f"    - Alta confianza: {high_conf}")
    print(f"    - Media confianza: {medium_conf}")
    print(f"    - Baja confianza: {low_conf}")
    if unknown > 0:
        print(f"    - Desconocidas: {unknown}")
    
    # Mostrar resultados
    print("\n[4] Mostrando resultados...")
    detector.display_results()


if __name__ == "__main__":
    main()
