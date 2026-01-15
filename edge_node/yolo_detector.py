import torch
import sys
from pathlib import Path

YOLO_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo"
sys.path.append(str(YOLO_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLODetector:
    def __init__(self, weights_path, conf=0.4):
        self.device = select_device("")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame):
        img = torch.from_numpy(frame).to(self.device)
        img = img.permute(2, 0, 1).contiguous().float()
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            preds = self.model(img)

        preds = non_max_suppression(preds, self.conf, 0.45)[0]

        people = []
        if preds is not None:
            preds[:, :4] = scale_boxes(img.shape[2:], preds[:, :4], frame.shape).round()
            for *box, conf, cls in preds:
                if self.names[int(cls)] == "person":
                    people.append(box)

        return people
