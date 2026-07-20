from ultralytics import YOLO
import torch
import logging
logger = logging.getLogger(__name__)


class ConeDetector:
   def __init__(self, config):
       self.config = config
       self.model = None
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.class_id_to_name = {int(class_id): name for class_id, name in self.config.class_names.items()}
       logger.info(f"Загрузка модели YOLO с {self.config.yolo_model_path} на {self.device}")
       try:
           self.model = YOLO(self.config.yolo_model_path)
           if hasattr(self.model, 'names'):
               logger.info(f"Классы модели: {self.model.names}")
       except Exception as e:
           logger.error(f"Ошибка загрузки модели: {e}")
           self.model = None
   
   def detect(self, frame):
       if self.model is None:
           return []
       try:
           with torch.inference_mode():
               results = self.model(
                   frame,
                   conf=self.config.confidence_threshold,
                   iou=self.config.iou_threshold,
                   verbose=False,
                   device=self.device,
               )

           detections = []

           for result in results:
               if result.boxes is None:
                   continue

               for box in result.boxes:
                   x1, y1, x2, y2 = map(int, box.xyxy[0])
                   conf = float(box.conf[0])
                   cls_id = int(box.cls[0])
                   cone_name = self.class_id_to_name.get(cls_id)

                   if cone_name is None:
                       continue

                   center_x = (x1 + x2) // 2
                   center_y = max(0, int(y1 + (y2 - y1) * self.config.point_of_view_offset_y))

                   detections.append({
                       'bbox': (x1, y1, x2, y2),
                       'conf': conf,
                       'class': cls_id,
                       'name': cone_name,
                       'center': (center_x, center_y)
                   })
           return detections
       except Exception as e:
           logger.debug(f"Ошибка детекции: {e}")
           return []
