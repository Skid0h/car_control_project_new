import logging
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger = logging.getLogger(__name__)

class ConeDetector:
    def __init__(self, config):
        self.config = config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
        logger.info(f"Инициализация чистого TensorRT: {self.config.yolo_model_path}")
        
        try:
            with open(self.config.yolo_model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Выделение памяти под I/O буферы видеокарты
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream()
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки TensorRT Engine: {e}")

    def detect(self, frame):
        if self.engine is None:
            return []

        # 1. Применяем ROI (обрезаем кадр)
        orig_h, orig_w = frame.shape[:2]
        crop_y1 = int(orig_h * getattr(self.config, 'roi_crop_top', 0.4))
        crop_y2 = int(orig_h * getattr(self.config, 'roi_crop_bottom', 1.0))
        cropped = frame[crop_y1:crop_y2, :]
        crop_h, crop_w = cropped.shape[:2]

        # 2. Подготовка изображения под размер YOLO (640x640)
        img_size = getattr(self.config, 'yolo_img_size', 640)
        img_resized = cv2.resize(cropped, (img_size, img_size))
        
        # BGR -> RGB, HWC -> CHW, нормализация 0-1
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0) # Добавляем batch_size
        
        # Копируем картинку в память (RAM -> GPU)
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Инференс (Выполнение нейросети)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Забираем результат (GPU -> RAM)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 4. Постобработка (YOLOv8 выдает [1, N_classes + 4, 8400])
        num_classes = len(self.config.class_names)
        channels = 4 + num_classes

        out = self.outputs[0]['host'].reshape(1, channels, 8400)
        out = out[0].T  # Транспонируем в (8400, channels)
        
        boxes = []
        scores = []
        class_ids = []
        
        # Масштаб для возврата координат к обрезанному кадру
        x_scale = crop_w / img_size
        y_scale = crop_h / img_size
        conf_thresh = getattr(self.config, 'confidence_threshold', 0.5)

        for row in out:
            cls_scores = row[4:channels] 
            class_id = np.argmax(cls_scores)
            score = cls_scores[class_id]
            
            if score > conf_thresh:
                # YOLO выдает: center_x, center_y, width, height
                cx, cy, w, h = row[0:4]
                
                # Переводим в координаты обрезанного кадра
                cx = cx * x_scale
                cy = cy * y_scale
                w = w * x_scale
                h = h * y_scale
                
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                
                boxes.append([x1, y1, int(w), int(h)])
                scores.append(float(score))
                class_ids.append(class_id)
                
        # 5. Non-Maximum Suppression (Удаляем дублирующиеся рамки)
        iou_thresh = getattr(self.config, 'iou_threshold', 0.5)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        
        detections = []
        if len(indices) > 0:
            name_to_id = {name: int(id_str) for id_str, name in self.config.class_names.items()}
            id_to_name = {v: k for k, v in name_to_id.items()}
            
            for i in np.array(indices).flatten():
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                
                # Возвращаем Y-координаты к полному 720p кадру (учитывая ROI)
                orig_y1 = y1 + crop_y1
                orig_y2 = y2 + crop_y1
                
                center_x = (x1 + x2) // 2
                pov_offset = getattr(self.config, 'point_of_view_offset_y', 0.8)
                center_y = int(orig_y1 + (orig_y2 - orig_y1) * pov_offset)
                
                cls_id = class_ids[i]
                cone_name = id_to_name.get(cls_id, "unknown")
                
                detections.append({
                    'bbox': (x1, orig_y1, x2, orig_y2),
                    'conf': scores[i],
                    'class': cls_id,
                    'name': cone_name,
                    'center': (center_x, center_y)
                })
                
        return detections
