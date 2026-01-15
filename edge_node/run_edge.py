import cv2
from edge_node.video_stream import VideoStream
from edge_node.yolo_detector import YOLODetector
from edge_node.anomaly_detector import AnomalyDetector
from edge_node.feature_extractor import CrowdFeatureExtractor

video = VideoStream(0)
detector = YOLODetector("models/yolo/weights/yolov5s.pt")
detector_ai = AnomalyDetector("models/edge/anomaly_model.pt")
extractor = CrowdFeatureExtractor()

while True:
    ret, frame = video.read()
    if not ret:
        break

    people = detector.detect(frame)

    features = extractor.extract(frame, people)
    score, is_anomaly = detector_ai.predict(features)
    if is_anomaly:
        cv2.putText(frame, "ANOMALY", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)


    for box in people:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Count: {features['count']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    print(score, is_anomaly)

    cv2.imshow("CrowdGuard Edge Node", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
