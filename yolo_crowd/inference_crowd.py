import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import random
import numpy as np
from utils.letterbox import letterbox
from config.get_config import get_inference_config

class CrowdInference:
    def __init__(self, model_path='weights/yolo-crowd.pt'):
        self.model_path = model_path
        self.device = select_device(get_inference_config()['device'])
        self.imgsz = get_inference_config()['imgsz']
        self.conf_thres = get_inference_config()['conf_thres']
        self.iou_thres = get_inference_config()['iou_thres']
        self.classes = get_inference_config()['classes']
        self.agnostic_nms = get_inference_config()['agnostic_nms']


    def load_model(self):
        self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        
        return self.model

    def load_image(self, image_path):
        img0 = cv2.imread(image_path)  # BGR
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.float()  # uint8 to fp16/32
        tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        return img, img0, tensor


    def postprocess(self, pred, im0, img):
        for i, det in enumerate(pred):  # detections per image
            if not len(det):
                continue
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):  # Loop through detections
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, 
                             label=label, 
                             color=self.colors[int(cls)], 
                             line_thickness=3)
        return im0

    def inference(self, image_path):
        img, img0, tensor = self.load_image(image_path)
        pred = self.model(tensor, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                   classes=self.classes, agnostic=self.agnostic_nms)
        visualized_image = self.postprocess(pred, img0, tensor)
        return visualized_image

if __name__ == '__main__':
    crowd_inference = CrowdInference()
    crowd_inference.load_model()
    visualized_image = crowd_inference.inference('data/bus.jpg')
    cv2.imshow('Visualized Image', visualized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()