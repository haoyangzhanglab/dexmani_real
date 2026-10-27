from fractions import Fraction
from pathlib import Path

import av


class VideoWriter:
    def __init__(self, path, *, fps, width, height):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.container = None
        self.stream = None
        self.frame_count = 0

    def open(self):
        self.container = av.open(str(self.path), "w", format="mp4")
        self.stream = self.container.add_stream("libx264", rate=self.rate)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv444p"
        self.stream.options = {"crf": "18", "preset": "ultrafast"}
        # 为控制和推理保留 CPU。
        self.stream.codec_context.thread_count = 1
        self.stream.codec_context.open()
        self.container.start_encoding()

    def write_frame(self, frame):
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        av_frame.pts = self.frame_count
        av_frame.time_base = 1 / self.rate
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.container is None:
            return
        try:
            if self.stream is not None:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
        finally:
            # 编码失败也关闭文件；关闭失败时保留句柄。
            self.stream = None
            self.container.close()
            self.container = None
