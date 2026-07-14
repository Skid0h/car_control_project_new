import logging
import cv2
import numpy as np
import tensorrt as trt           # NVIDIA TensorRT для инференса нейросетей
import pycuda.driver as cuda     # Управление памятью видеокарты напрямую
import pycuda.autoinit           # Автоматически инициализирует CUDA-контекст

logger = logging.getLogger(__name__)

class ConeDetector:
    def __init__(self, config):
        self.config = config
        # Настраиваем логгер TensorRT, чтобы он не спамил инфо-сообщениями
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
        logger.info(f"Инициализация чистого TensorRT: {self.config.yolo_model_path}")
        
        try:
            # Десериализация (распаковка) скомпилированной модели .engine
            with open(self.config.yolo_model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            # Создаем контекст выполнения (в нем будут храниться активации нейросети во время работы)
            self.context = self.engine.create_execution_context()
            
            # Списки для хранения ссылок на выделенную память
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream() # Создаем асинхронный поток CUDA
            
            # ==========================================
            # ВЫДЕЛЕНИЕ ПАМЯТИ (RAM -> GPU)
            # ==========================================
            # Проходим по всем входам (картинка) и выходам (координаты рамок) сети
            for binding in self.engine:
                # Считаем нужный объем памяти: перемножаем размерности тензора на максимальный размер батча
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # pagelocked_empty выделяет "закрепленную" память в RAM (не уходит в своп).
                # Это позволяет копировать данные в GPU (VRAM) в разы быстрее, чем через обычный numpy массив.
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes) # Выделяем аналогичный блок в памяти самой видеокарты
                
                self.bindings.append(int(device_mem))
                
                # Сортируем: что-то пойдет на вход, что-то будет результатом на выходе
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки TensorRT Engine: {e}")

    def detect(self, frame):
        """Главная функция обработки кадра. Принимает картинку (numpy array), возвращает список конусов."""
        if self.engine is None:
            return []

        # ==========================================
        # 1. ПОДГОТОВКА ИЗОБРАЖЕНИЯ (PREPROCESSING)
        # ==========================================
        # Применяем ROI: обрезаем небо и края, где конусов точно не может быть (экономит ресурсы)
        orig_h, orig_w = frame.shape[:2]
        crop_y1 = int(orig_h * getattr(self.config, 'roi_crop_top', 0.3))
        crop_y2 = int(orig_h * getattr(self.config, 'roi_crop_bottom', 1.0))
        cropped = frame[crop_y1:crop_y2, :]
        crop_h, crop_w = cropped.shape[:2]

        # Ресайз до квадратного размера (640x640), который ждет YOLO
        img_size = getattr(self.config, 'yolo_img_size', 640)
        img_resized = cv2.resize(cropped, (img_size, img_size))
        
        # Конвертация
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # HWC (высота-ширина-каналы) -> CHW (каналы-высота-ширина)
        # Деление на 255.0 - нормализация пикселей 
        img_tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        # Добавляем размерность батча: (3, 640, 640) -> (1, 3, 640, 640)
        img_tensor = np.expand_dims(img_tensor, axis=0) 
        
        # ==========================================
        # 2. ИНФЕРЕНС (ОТПРАВКА В ВИДЕОКАРТУ)
        # ==========================================
        # Плоско (ravel) копируем наш тензор в выделенную область оперативной памяти
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        # Асинхронно перебрасываем данные по шине PCIe в память видеокарты
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Запуск самой нейросети
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Забираем результат инференса обратно из видеокарты в RAM
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        # Синхронизация потока: ждем, пока видеокарта закончит работу
        self.stream.synchronize()
        
        # ==========================================
        # 3. ПОСТОБРАБОТКА (POSTPROCESSING)
        # ==========================================
        num_classes = len(self.config.class_names)
        channels = 4 + num_classes
        
        # Превращаем одномерный массив обратно в матрицу [1, channels, 8400]
        out = self.outputs[0]['host'].reshape(1, channels, 8400)
        # Транспонируем матрицу, чтобы удобнее было итерироваться по 8400 найденным гипотетическим рамкам
        out = out[0].T  
        
        boxes = []
        scores = []
        class_ids = []
        
        # Коэффициенты масштабирования, чтобы вернуть рамки из размера 640x640 к размеру обрезанного кадра
        x_scale = crop_w / img_size
        y_scale = crop_h / img_size
        conf_thresh = getattr(self.config, 'confidence_threshold', 0.5)

        for row in out:
            # Срез от 4 до конца (вероятности принадлежности рамки к каждому из классов)
            cls_scores = row[4:channels] 
            class_id = np.argmax(cls_scores) # Выбираем класс с максимальной вероятностью
            score = cls_scores[class_id]     # Получаем саму цифру вероятности
            
            if score > conf_thresh: # Отбрасываем мусор (уверенность < 50%)
                # YOLO выдает 4 цифры: координаты центра рамки, ее ширину и высоту
                cx, cy, w, h = row[0:4]
                
                # Масштабируем эти цифры обратно под обрезанный кадр камеры
                cx = cx * x_scale
                cy = cy * y_scale
                w = w * x_scale
                h = h * y_scale
                
                # Вычисляем левый верхний угол рамки (YOLO выдает центр)
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                
                boxes.append([x1, y1, int(w), int(h)])
                scores.append(float(score))
                class_ids.append(class_id)
                
        # ==========================================
        # 4. ФИЛЬТРАЦИЯ (NON-MAXIMUM SUPPRESSION)
        # ==========================================
        # NMS убирает дубликаты. Нейросеть часто выдает 5-10 перекрывающихся рамок на один реальный конус.
        iou_thresh = getattr(self.config, 'iou_threshold', 0.5) # Насколько рамки должны перекрываться, чтобы считаться дубликатом
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        
        detections = []
        if len(indices) > 0:
            # Создаем словари для перевода ID (0,1,2) в человекочитаемые имена (yellow, blue...)
            name_to_id = {name: int(id_str) for id_str, name in self.config.class_names.items()}
            id_to_name = {v: k for k, v in name_to_id.items()}
            
            # --- ИСПРАВЛЕНИЕ ---
            # Оборачиваем indices в np.array() для защиты от старых версий OpenCV, где может возвращаться пустой tuple
            for i in np.array(indices).flatten():
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                
                # Восстанавливаем оригинальные Y-координаты. 
                # Так как мы в начале отрезали 30% кадра (crop_y1), нужно прибавить эту высоту обратно.
                orig_y1 = y1 + crop_y1
                orig_y2 = y2 + crop_y1
                
                center_x = (x1 + x2) // 2
                
                # Для конусов вычисляем центр не посередине рамки, а ближе к земле (базе конуса).
                # Это нужно, чтобы карта глубины (depth map) бралась с тела конуса, а не с пустоты/верхушки.
                pov_offset = getattr(self.config, 'point_of_view_offset_y', 0.8)
                center_y = int(orig_y1 + (orig_y2 - orig_y1) * pov_offset)
                
                cls_id = class_ids[i]
                cone_name = id_to_name.get(cls_id, "unknown")
                
                # Формируем итоговый словарь для сервера
                detections.append({
                    'bbox': (x1, orig_y1, x2, orig_y2), # Координаты рамки на оригинальном 720p кадре
                    'conf': scores[i],                  # Уверенность 0.0 - 1.0
                    'class': cls_id,                    # ID класса
                    'name': cone_name,                  # Имя (yellow, blue, orange)
                    'center': (center_x, center_y)      # Точка для запроса к карте глубины ZED
                })
                
        return detections
