import cv2

class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera source {source} not accessible")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
