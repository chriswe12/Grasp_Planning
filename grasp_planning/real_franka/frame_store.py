"""Bounded nonblocking frame queue to a private local spool; no encoding in control."""

import pickle
import queue
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np


class FrameStore:
    def __init__(self, folder):
        file = tempfile.NamedTemporaryFile(prefix="franka-capture-", suffix=".tmp", dir=folder, delete=False)
        self.path = Path(file.name)
        self.queue = queue.Queue(maxsize=32)
        self.error = None
        self.count = 0
        self.sealed = False

        def write():
            try:
                with file:
                    while True:
                        item = self.queue.get()
                        try:
                            if item is None:
                                return
                            pickle.dump(item, file, protocol=5)
                        finally:
                            self.queue.task_done()
            except Exception as exc:
                self.error = exc

        self.worker = threading.Thread(target=write, name="franka-frame-spool", daemon=True)
        self.worker.start()

    def __len__(self):
        return self.count

    def append(self, item):
        if self.error:
            raise RuntimeError(f"Recording storage failed: {self.error}")
        if self.sealed:
            raise RuntimeError("Recording already sealed")
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("Recording storage cannot keep up; stopping rather than blocking control") from exc
        self.count += 1

    def seal(self):
        if not self.sealed:
            while self.worker.is_alive():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.worker.join()
            self.sealed = True
        if self.error:
            raise RuntimeError(f"Recording storage failed: {self.error}; spool retained at {self.path}")

    def __iter__(self):
        self.seal()
        # Only our private, freshly created spool is read, never a supplied pickle.
        with self.path.open("rb") as file:
            for _ in range(self.count):
                yield pickle.load(file)

    def cleanup(self):
        self.seal()
        self.path.unlink(missing_ok=True)


def save_rgbd_archive(path, frames, goal):
    """Write standard NPZ arrays one frame at a time, avoiding a full-run stack."""
    first = next(iter(frames))
    fields = {
        "live_rgbd": lambda f: np.asarray(f[1]),
        "time_s": lambda f: np.asarray(f[2]["time_s"], dtype=np.float64),
        "frame_seq": lambda f: np.asarray(f[2]["frame_seq"], dtype=np.int64),
        "raw_depth_m": lambda f: np.asarray(f[3]),
    }
    partial = path.with_suffix(".partial.npz")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        with archive.open("goal_rgbd.npy", "w", force_zip64=True) as out:
            np.lib.format.write_array(out, goal, allow_pickle=False)
        for name, extract in fields.items():
            example = extract(first)
            with archive.open(name + ".npy", "w", force_zip64=True) as out:
                np.lib.format.write_array_header_2_0(
                    out,
                    dict(
                        descr=np.lib.format.dtype_to_descr(example.dtype),
                        fortran_order=False,
                        shape=(len(frames), *example.shape),
                    ),
                )
                for frame in frames:
                    value = extract(frame)
                    if value.shape != example.shape or value.dtype != example.dtype:
                        raise RuntimeError(f"Recording array format changed: {name}")
                    out.write(value.tobytes(order="C"))
    partial.replace(path)
