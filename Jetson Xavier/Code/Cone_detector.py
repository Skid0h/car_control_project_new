import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ConeDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        
        # Жестко задаем GPU 0 для TensorRT на Jetson
        self.device = 0 
        
        logger.info(f"Загрузка модели YOLO (TensorRT) с {self.config.yolo_model_path} на устройство: GPU {self.device}")
        
        try:
            self.model = YOLO(self.config.yolo_model_path)
            if hasattr(self.model, 'names'):
                logger.info(f"Классы модели YOLO: {self.model.names}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели YOLO: {e}")
            self.model = None

        # ==========================================================
        # ОПТИМИЗАЦИЯ 1: Кэшируем словари классов ОДИН РАЗ
        # ==========================================================
        # Раньше они пересоздавались каждый кадр, нагружая CPU
        self.id_to_name = {int(k): v for k, v in self.config.class_names.items()}
        self.conf_thresh = getattr(self.config, 'confidence_threshold', 0.5)
        self.iou_thresh = getattr(self.config, 'iou_threshold', 0.5)
        self.img_size = getattr(self.config, 'yolo_img_size', 640)
        self.roi_top = getattr(self.config, 'roi_crop_top', 0.3)
        self.roi_bottom = getattr(self.config, 'roi_crop_bottom', 1.0)
        self.pov_offset = getattr(self.config, 'point_of_view_offset_y', 0.8)

    def detect(self, frame):
        if self.model is None:
            return []
            
        try:
            # ==========================================
            # 1. ПОДГОТОВКА ИЗОБРАЖЕНИЯ (ROI CROP)
            # ==========================================
            orig_h, orig_w = frame.shape[:2]
            crop_y1 = int(orig_h * self.roi_top)
            crop_y2 = int(orig_h * self.roi_bottom)
            cropped = frame[crop_y1:crop_y2, :]
            
            # ==========================================
            # 2. ИНФЕРЕНС (ОПТИМИЗИРОВАН)
            # ==========================================
            # half=True включает FP16 (на Jetson TRT это дает +50-100% к скорости)
            # imgsz гарантирует, что сеть не будет гадать с размером
            results = self.model(
                cropped,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,
                device=self.device,
                imgsz=self.img_size,
                half=True 
            )
            
            # ==========================================
            # 3. ПОСТОБРАБОТКА И МАСШТАБИРОВАНИЕ
            # ==========================================
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Берем из кэша, а не пересоздаем словарь
                        cone_name = self.id_to_name.get(cls_id, None)
                        if cone_name is None:
                            continue 
                            
                        orig_y1 = y1 + crop_y1
                        orig_y2 = y2 + crop_y1
                        center_x = (x1 + x2) // 2
                        center_y = int(orig_y1 + (orig_y2 - orig_y1) * self.pov_offset)
                        
                        detections.append({
                            'bbox': (x1, orig_y1, x2, orig_y2),
                            'conf': conf,
                            'class': cls_id,
                            'name': cone_name,
                            'center': (center_x, center_y)
                        })
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка во время детекции конусов: {e}")
            return []
