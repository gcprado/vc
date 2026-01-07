import torch
import open_clip
from PIL import Image

class CLIPClassifier:
    def __init__(self, model_name="ViT-B-32", checkpoint_path=None, device=None):
        """
        Inicializa un modelo CLIP para clasificación zero-shot.

        Args:
            model_name (str): nombre del modelo CLIP a usar.
            checkpoint_path (str, optional): ruta al checkpoint local (.safetensors). Si None, usa pesos preentrenados.
            device (str, optional): 'cuda' o 'cpu'. Si None, se selecciona automáticamente.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Cargar el modelo y el preprocesador
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Cargar el checkpoint si se proporciona
        if checkpoint_path is not None:
            open_clip.load_checkpoint(self.model, checkpoint_path, device=self.device)
            print(f"✅ CLIP model loaded from checkpoint: {checkpoint_path}")
        else:
            print(f"✅ CLIP model loaded with pretrained weights ({model_name})")

        # Pasar el modelo al dispositivo y ponerlo en modo evaluación
        self.model.to(self.device)
        self.model.eval()

    def classify(self, image, class_names):
        """
        Clasifica un ROI usando zero-shot CLIP.

        Args:
            image (np.ndarray): imagen recortada (BGR o RGB) como array de numpy.
            class_names (list[str]): lista de nombres de clases candidatos.

        Returns:
            str: nombre de la clase más probable
        """
        # Convertir ROI a PIL 
        image_pil = Image.fromarray(image)

        # Preprocesar imagen
        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # Preparar textos candidatos
        texts = [f"a photo of {product}" for product in class_names]
        text_tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            # Extraer embeddings
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)

            # Normalizar embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calcular similitud coseno simple
            similarity = (image_features @ text_features.T).squeeze(0)

            # Calcular similitud coseno con probabilidades para topk y visualizar confianza 
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Obtener índice del texto más similar, obener scores y k con topk 
        best_idx = similarity.argmax().item()
        predicted_class = class_names[best_idx]

        return predicted_class