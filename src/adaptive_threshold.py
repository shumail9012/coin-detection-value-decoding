class AdaptiveResolutionController:
    def __init__(
        self,
        target_fps=20,
        min_width=640,
        max_width=1280,
        step=160,
        settle_frames=15,
        tolerance=0.15
    ):
        self.target_fps = target_fps
        self.min_width = min_width
        self.max_width = max_width
        self.step = step
        self.settle_frames = settle_frames
        self.tolerance = tolerance

        self.current_width = max_width
        self.counter = 0

    def update(self, measured_fps):
        lower = self.target_fps * (1 - self.tolerance)
        upper = self.target_fps * (1 + self.tolerance)

        self.counter += 1
        if self.counter < self.settle_frames:
            return self.current_width

        self.counter = 0

        if measured_fps < lower:
            self.current_width = max(
                self.min_width,
                self.current_width - self.step
            )

        elif measured_fps > upper:
            self.current_width = min(
                self.max_width,
                self.current_width + self.step
            )

        return self.current_width
