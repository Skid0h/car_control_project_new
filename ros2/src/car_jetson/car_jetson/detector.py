"""TensorRT 8 binding API for JetPack 5.1.3. No PyTorch dependency."""
from pathlib import Path
import threading
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


def output_rows(output, classes):
    columns = 4 + classes
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError('Expected raw YOLOv8/v11 [1,4+C,N] or [1,N,4+C], no embedded NMS')
    result = output[0]
    if result.shape[0] == columns:
        result = result.T
    elif result.shape[1] != columns:
        raise ValueError('Engine class count does not match config; YOLOv5/COCO/NMS output unsupported')
    return result


class ConeDetector:
    def __init__(self, config):
        self.config = config
        self.owner = threading.get_ident()
        self.cuda_ctx = None
        self.engine = self.context = self.runtime = self.stream = None
        self.buffers = []
        self.class_id_to_name = {int(k): v for k, v in config.class_names.items()}
        if sorted(self.class_id_to_name) != list(range(len(self.class_id_to_name))):
            raise ValueError('Class IDs must be consecutive from zero')
        if trt.__version__.split('.')[0] != '8':
            raise RuntimeError('This detector requires TensorRT 8.x from JetPack 5')
        self.preprocess = getattr(config, 'preprocess', 'stretch')
        if self.preprocess not in ('stretch', 'letterbox'):
            raise ValueError('preprocess must be stretch or letterbox')
        cuda.init()
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        try:
            self.cuda_ctx.push()
            try:
                self.logger = trt.Logger(trt.Logger.WARNING)
                self.runtime = trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(Path(config.yolo_model_path).read_bytes())
                if self.engine is None:
                    raise RuntimeError('Engine cannot be loaded; build it on Xavier with TensorRT 8.5.2')
                if self.engine.has_implicit_batch_dimension or self.engine.num_bindings != 2:
                    raise ValueError('Expected explicit batch engine with exactly one input and one raw output')
                inputs = [i for i in range(2) if self.engine.binding_is_input(i)]
                if len(inputs) != 1:
                    raise ValueError('Expected one input')
                self.input_index = inputs[0]
                self.output_index = 1 - self.input_index
                shape = tuple(self.engine.get_binding_shape(self.input_index))
                if len(shape) != 4 or shape[0] not in (1, -1) or shape[1] not in (3, -1):
                    raise ValueError('Expected NCHW batch=1 RGB input')
                # Dynamic exports use the established 640-square model profile.
                desired = tuple(fixed if fixed > 0 else default for fixed, default in zip(shape, (1, 3, 640, 640)))
                self.height, self.width = desired[2:]
                self.context = self.engine.create_execution_context()
                if self.context is None:
                    raise RuntimeError('TensorRT execution context failed')
                if -1 in shape and not self.context.set_binding_shape(self.input_index, desired):
                    raise ValueError('Engine profile does not accept 1x3x640x640')
                self.bindings = [0, 0]
                self.stream = cuda.Stream()
                for i in range(2):
                    resolved = tuple(self.context.get_binding_shape(i))
                    if any(d <= 0 for d in resolved):
                        raise ValueError('Unresolved dynamic binding: ' + str(resolved))
                    dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
                    if dtype not in (np.dtype('float32'), np.dtype('float16')):
                        raise ValueError('Only FP32/FP16 input/output supported')
                    if self.engine.get_binding_format(i) != trt.TensorFormat.LINEAR:
                        raise ValueError('Only LINEAR tensor bindings supported')
                    host = cuda.pagelocked_empty(int(np.prod(resolved)), dtype)
                    device = cuda.mem_alloc(host.nbytes)
                    self.buffers.append(dict(host=host, device=device, shape=resolved))
                    self.bindings[i] = int(device)
                output_rows(self.buffers[self.output_index]['host'].reshape(
                    self.buffers[self.output_index]['shape']), len(self.class_id_to_name))
            finally:
                self.cuda_ctx.pop()
        except BaseException:
            self.close()
            raise

    def detect(self, frame):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Create, infer and close detector on the same thread')
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError('Empty frame')
        if self.preprocess == 'letterbox':
            ratio = min(self.width / w, self.height / h)
            rw, rh = round(w * ratio), round(h * ratio)
            left, top = (self.width - rw) // 2, (self.height - rh) // 2
            img = cv2.copyMakeBorder(cv2.resize(frame, (rw, rh)), top, self.height-rh-top,
                                    left, self.width-rw-left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            sx, sy = rw / w, rh / h
        else:
            img = cv2.resize(frame, (self.width, self.height))
            sx, sy, left, top = self.width / w, self.height / h, 0, 0
        inp = self.buffers[self.input_index]
        out = self.buffers[self.output_index]
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        np.copyto(inp['host'], (data.astype(inp['host'].dtype) / 255).ravel())
        self.cuda_ctx.push()
        try:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            if not self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle):
                raise RuntimeError('TensorRT inference failed')
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
        finally:
            self.cuda_ctx.pop()
        rows = output_rows(out['host'].reshape(out['shape']), len(self.class_id_to_name))
        rows = rows[np.isfinite(rows).all(axis=1)]
        classes = np.argmax(rows[:, 4:], axis=1)
        scores = np.max(rows[:, 4:], axis=1)
        keep = (scores > self.config.confidence_threshold) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
        rows, classes, scores = rows[keep], classes[keep], scores[keep]
        boxes = [[float((cx-bw/2-left)/sx), float((cy-bh/2-top)/sy), float(bw/sx), float(bh/sy)]
                 for cx, cy, bw, bh in rows[:, :4]]
        selected = []
        for cls in np.unique(classes):
            indices = np.flatnonzero(classes == cls)
            kept = cv2.dnn.NMSBoxes([boxes[i] for i in indices], scores[indices].astype(float).tolist(),
                                    self.config.confidence_threshold, self.config.iou_threshold)
            if len(kept):
                selected.extend(indices[np.asarray(kept).reshape(-1)])
        result = []
        for i in selected:
            x, y, bw, bh = boxes[i]
            x1, y1, x2, y2 = max(0, round(x)), max(0, round(y)), min(w, round(x+bw)), min(h, round(y+bh))
            if x2 > x1 and y2 > y1:
                result.append(dict(bbox=(x1,y1,x2,y2), center=((x1+x2)//2,(y1+y2)//2),
                                   conf=float(scores[i]), name=self.class_id_to_name[int(classes[i])]))
        return result

    def close(self):
        if self.cuda_ctx is None:
            return
        if threading.get_ident() != self.owner:
            raise RuntimeError('Close detector on its owning thread')
        self.cuda_ctx.push()
        try:
            if self.stream is not None:
                self.stream.synchronize()
            self.context = None
            for buffer in self.buffers:
                buffer['device'].free()
            self.buffers.clear()
            self.stream = self.engine = self.runtime = None
        finally:
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
            self.cuda_ctx = None
