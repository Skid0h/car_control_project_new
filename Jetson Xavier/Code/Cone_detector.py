import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ConeDetector:
    def __init__(self, config):
        self.config = config
        self.class_id_to_name = {int(k): v for k, v in self.config.class_names.items()}
        
        # Задаем размер, под который был скомпилирован .engine файл
        self.imgsz = 640 
        self.cuda_ctx = None
        
        logger.info(f"Загрузка нативного TensorRT engine: {self.config.yolo_model_path}")
        
        try:
            # 1. Инициализируем драйвер CUDA
            try:
                cuda.init()
            except cuda.LogicError:
                pass # Уже инициализировано
            
            self.dev = cuda.Device(0)
            self.cuda_ctx = self.dev.make_context()
            
            # 2. КРИТИЧЕСКИ ВАЖНО: Пушим контекст ПЕРЕД созданием execution context!
            # Это привязывает внутренние ресурсы TensorRT именно к нашему PyCUDA контексту.
            self.cuda_ctx.push()
            
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.engine = self._load_engine(self.config.yolo_model_path)
            
            # 3. Создаем execution context (теперь он валиден)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine)
            
            # 4. Убираем контекст из стека главного потока
            self.cuda_ctx.pop()
            
            logger.info("TensorRT Runtime успешно инициализирован!")
        except Exception as e:
            logger.error(f"Ошибка загрузки TRT Engine: {e}")
            self.engine = None
            if self.cuda_ctx:
                try: self.cuda_ctx.pop()
                except: pass

    def __del__(self):
        """Гарантированное очищение стека контекстов при завершении программы"""
        if self.cuda_ctx:
            try:
                self.cuda_ctx.synchronize()
                self.cuda_ctx.pop()
            except:
                pass
            try:
                self.cuda_ctx.detach()
            except:
                pass

    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            size = trt.volume(engine.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': engine.get_binding_shape(i)})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': engine.get_binding_shape(i)})
        return inputs, outputs, bindings, stream

    def detect(self, frame):
        if self.engine is None or frame is None:
            return []
            
        orig_h, orig_w = frame.shape[:2]
        
        # ==========================================
        # 1. ПРЕ-ПРОЦЕССИНГ (CPU)
        # ==========================================
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        np.copyto(self.inputs[0]['host'], img.ravel())
        
        # ==========================================
        # 2. ИНФЕРЕНС (GPU)
        # ==========================================
        # Пушим контекст в рабочем потоке
        self.cuda_ctx.push()
        try:
            # Для explicit batch engine ОБЯЗАТЕЛЬНО задаем shape
            self.context.set_binding_shape(0, (1, 3, self.imgsz, self.imgsz))
            
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
        except Exception as e:
            logger.error(f"Ошибка инференса TRT: {e}")
        finally:
            # Всегда убираем контекст, даже если была ошибка
            self.cuda_ctx.pop()

        # ==========================================
        # 3. ПОСТ-ПРОЦЕССИНГ (CPU)
        # ==========================================
        output_data = self.outputs[0]['host']
        out_shape = self.outputs[0]['shape'] 
        
        output_data = output_data.reshape(out_shape) 
        output_data = output_data[0].T 
        
        boxes_raw = output_data[:, :4]
        scores_raw = output_data[:, 4:]
        class_ids = np.argmax(scores_raw, axis=1)
        confidences = np.max(scores_raw, axis=1)
        
        mask = confidences > self.config.confidence_threshold
        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes_raw) == 0:
            return []
            
        x_scale = orig_w / self.imgsz
        y_scale = orig_h / self.imgsz
        boxes_nms = []
        for (cx, cy, w, h) in boxes_raw:
            x = (cx - w / 2) * x_scale
            y = (cy - h / 2) * y_scale
            bw = w * x_scale
            bh = h * y_scale
            boxes_nms.append([int(x), int(y), int(bw), int(bh)])
            
        indices = cv2.dnn.NMSBoxes(
            boxes_nms, 
            confidences.tolist(), 
            self.config.confidence_threshold, 
            self.config.iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes_nms[i]
                conf = float(confidences[i])
                cls_id = int(class_ids[i])
                cone_name = self.class_id_to_name.get(cls_id)
                if cone_name is None: 
                    continue
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(orig_w, int(x + bw)), min(orig_h, int(y + bh))
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
