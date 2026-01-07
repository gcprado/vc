import os
import re
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class BatchYOLOReporter:
    def __init__(self, detector, output_dir="output"):
        self.detector = detector
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def snapshot_invetory(self, image_dir: str):
        """
        Procesa todas las imágenes del directorio:
        - Crea carpeta output/<run_timestamp>/
        - Corre YOLO
        - Dibuja boxes
        - Guarda imágenes con nombre:
        <SECCION>_<COUNT>.jpg
        """

        # -------------------------------
        # Timestamp global del run
        # -------------------------------
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_output_dir = os.path.join(self.output_dir, run_timestamp)
        os.makedirs(run_output_dir, exist_ok=True)

        # -------------------------------
        # Listar imágenes
        # -------------------------------
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_files.sort()

        print(f"Procesando {len(image_files)} imágenes...")

        # Lista para guardar datos CSV
        csv_data = []

        # -------------------------------
        # Loop principal
        # -------------------------------
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Inferencia
            boxes, _ = self.detector.detect(image)
            img_with_boxes = self.detector.draw_boxes(image.copy(), boxes)

            num_objects = len(boxes)

            # -------------------------------
            # Nombre de salida
            # -------------------------------
            base_name = os.path.splitext(img_file)[0]
            output_name = f"{base_name}_{num_objects}.jpg"

            out_path = os.path.join(run_output_dir, output_name)

            # Guardar imagen
            img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, img_bgr)

            section_name = re.sub(r'[-_]*\d+$', '', base_name)  # quita -1 o _1 al final
            # Guardar datos CSV
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                csv_data.append({
                    "image_file": img_file,
                    "output_file": output_name,
                    "section": section_name,
                    "bbox:": bbox
                })
        
        # Guardar CSV
        csv_path = os.path.join(run_output_dir, f"bboxes_{run_timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        print("\n Proceso terminado.")

    def print_summary(self, output_dir):
        print(f"\n RESUMEN DE DETECCIONES DE {output_dir}")
        print("=" * 50)

        files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".jpg")])

        for f in files:
            name = os.path.splitext(f)[0]
            parts = name.split("_")

            section = parts[0]
            count = parts[1]
            timestamp = "_".join(parts[2:])

            print(f"{section:20s} -> {count:4s} productos")

    def show_summary(self, output_dir, cols=3):
        files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".jpg")])

        images = []
        titles = []

        for f in files:
            path = os.path.join(output_dir, f)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            titles.append(f)

        # Mostrar en grillas de 3
        for i in range(0, len(images), cols):
            batch_imgs = images[i:i+cols]
            batch_titles = titles[i:i+cols]

            plt.figure(figsize=(5 * len(batch_imgs), 5))

            for j, (img, title) in enumerate(zip(batch_imgs, batch_titles)):
                plt.subplot(1, len(batch_imgs), j + 1)
                plt.imshow(img)
                plt.title(title, fontsize=9)
                plt.axis("off")

            plt.tight_layout()
            plt.show()

