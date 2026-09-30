"""One-owner ZED capture with an immutable latest frame and bounded freshness."""

import threading
import time

import numpy as np


class ZedCamera:
    def __init__(self, serial=13829658):
        self.serial = serial
        self.latest = None
        self.calibration = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        return self

    def _capture(self):
        import pyzed.sl as sl

        cam = sl.Camera()
        try:
            p = sl.InitParameters()
            p.set_from_serial_number(self.serial)
            p.camera_resolution = sl.RESOLUTION.VGA
            p.camera_fps = 30
            p.coordinate_units = sl.UNIT.METER
            p.depth_mode = sl.DEPTH_MODE.NEURAL
            p.depth_minimum_distance = 0.1
            p.depth_maximum_distance = 1.0
            status = cam.open(p)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED open: {status}")
            info = cam.get_camera_information()
            cfg = info.camera_configuration
            k = cfg.calibration_parameters.left_cam
            if info.camera_model != sl.MODEL.ZED_M:
                raise ValueError("Expected a ZED Mini")
            self.calibration = dict(
                serial=int(info.serial_number),
                model=str(info.camera_model),
                sdk=sl.Camera.get_sdk_version(),
                width=cfg.resolution.width,
                height=cfg.resolution.height,
                fx=float(k.fx),
                fy=float(k.fy),
                cx=float(k.cx),
                cy=float(k.cy),
                baseline_m=float(cfg.calibration_parameters.get_camera_baseline()),
                depth_mode="NEURAL",
                color="RGB",
                depth="float32 optical Z metres",
            )
            rgb = sl.Mat()
            depth = sl.Mat()
            seq = 0
            while not self.stop_event.is_set():
                status = cam.grab()
                if status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"ZED grab: {status}")
                stamp_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                cam.retrieve_image(rgb, sl.VIEW.LEFT)
                cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Preserve the capture age, including GPU processing before delivery.
                age = max(0.0, (sl.get_current_timestamp().get_nanoseconds() - stamp_ns) * 1e-9)
                seq += 1
                self.latest = (
                    time.monotonic() - age,
                    seq,
                    rgb.get_data()[..., [2, 1, 0]].copy(),
                    depth.get_data().astype(np.float32, copy=True),
                )
        except Exception as e:
            self.error = str(e)
        finally:
            cam.close()

    def frame(self, max_age=0.15):
        if self.error:
            raise RuntimeError(self.error)
        f = self.latest
        if f is None:
            raise RuntimeError("Waiting for camera")
        if time.monotonic() - f[0] > max_age:
            raise RuntimeError("Camera frame is stale")
        return f

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
