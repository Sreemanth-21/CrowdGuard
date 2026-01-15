import numpy as np
from edge_node.video_stream import VideoStream
from edge_node.yolo_detector import YOLODetector
from edge_node.feature_extractor import CrowdFeatureExtractor

video = VideoStream(0)
detector = YOLODetector("models/yolo/weights/yolov5s.pt")
extractor = CrowdFeatureExtractor()

features = []

for _ in range(500):
    ret, frame = video.read()
    if not ret:
        break

    boxes = detector.detect(frame)
    feat = extractor.extract(frame, boxes)

    features.append([
        feat["count"],
        feat["density"],
        feat["avg_distance"],
        feat["motion"],
        feat["pressure"]
    ])

video.release()

np.save("data/processed/normal_features.npy", np.array(features))
