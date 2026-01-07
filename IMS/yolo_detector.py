from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded from: {model_path}")


    def detect(self, image):
        """
        Detecta objetos en una imagen utilizando el modelo YOLO.

        Args:
            image (str | np.ndarray): path a la imagen o imagen ya cargada (RGB/BGR)
        
        Returns:
            boxes: bounding boxes detectadas por YOLO
            results: objeto completo de resultados de YOLO
        """
        results = self.model(image, verbose=False)
        boxes = results[0].boxes
        return boxes, results[0]
    
    def draw_boxes(self, image, boxes):
        """Dibuja los bounding boxes sobre la imagen"""
        img_draw = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"Obj {i+1}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return img_draw

