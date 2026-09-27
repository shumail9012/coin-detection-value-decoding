import threading
import time

class CameraProducer:
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.running:
            frame = self.camera.read()  # or grab Pylon frame
            if frame is not None:
                with self.lock:
                    self.frame = frame
            time.sleep(0.001)  # reduce busy wait, keeps ~1000 FPS loop


    def get_latest(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
