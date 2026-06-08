from ultralytics import YOLO
import torch
import logging
logger = logging.getLogger(__name__)


class ConeDetector:
   def __init__(self, config):
       self.detection = config['detection']
       self.vision = config['vision']
       self.model = None
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       logger.info(f"Загрузка модели YOLO с {self.vision['yolo_model_path']} на {self.device}")
       try:
           self.model = YOLO(self.vision['yolo_model_path'])
           if hasattr(self.model, 'names'):
               logger.info(f"Классы модели: {self.model.names}")
       except Exception as e:
           logger.error(f"Ошибка загрузки модели: {e}")
           self.model = None
   
   def detect(self, frame):
       if self.model is None:
           return []
       try:
           results = self.model(frame, conf=self.vision['confidence_threshold'], iou=self.vision['iou_threshold'], verbose=False, device=self.device)
           detections = []
           
           name_to_id = {name: int(id_str) for id_str, name in self.detection['class_names'].items()}
           
           for result in results:
               if result.boxes is not None:
                   for box in result.boxes:
                       x1, y1, x2, y2 = map(int, box.xyxy[0])
                       conf = float(box.conf[0])
                       cls_id = int(box.cls[0])
                       
                       cone_name = None
                       for name, cid in name_to_id.items():
                           if cid == cls_id:
                               cone_name = name
                               break
                       
                       if cone_name is None:
                           continue
                       
                       center_x = (x1 + x2) // 2
                       center_y = int(y1 + (y2 - y1) * self.vision['point_of_view_offset_y']) 
                       
                       detections.append({
                           'bbox': (x1, y1, x2, y2),
                           'conf': conf,
                           'class': cls_id,
                           'name': cone_name,
                           'center': (center_x, center_y)
                       })
           return detections
       except Exception as e:
           logger.error(f"Ошибка детекции: {e}")
           return []