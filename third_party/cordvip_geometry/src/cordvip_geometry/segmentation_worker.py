"""One-shot local Grounded-SAM process; no model or hardware imports in the UI.

The child exits before a mask becomes available for pose registration, releasing
all segmentation CUDA allocations. Cancelling retains its diagnostic files but
never publishes candidates from the cancelled source frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .live_workers import CloseStatus
from ._process import module_environment


@dataclass(frozen=True)
class SegmentationResult:
    generation: int
    candidates: list[dict]
    directory: Path
    error: str | None


class SegmentationWorker:
    def __init__(self, directory, *, dino_weights, sam_root, sam_checkpoint, sam_config,
                 box_threshold=.3, text_threshold=.25):
        self.directory = Path(directory)
        self.options = dict(dino_weights=str(Path(dino_weights).expanduser().resolve()),
                            sam_root=str(Path(sam_root).expanduser().resolve()),
                            sam_checkpoint=str(Path(sam_checkpoint).expanduser().resolve()),
                            sam_config=sam_config, box_threshold=box_threshold,
                            text_threshold=text_threshold)
        self._process = None
        self._log = None
        self._job = None
        self._index = 0
        self._cancelled_at = None
        self._closed = False

    @property
    def busy(self):
        # Even an exited child must be reaped before a new request can replace it.
        return self._process is not None

    def submit(self, rgb, prompt, *, generation, source_id, timestamp_ns):
        if self._closed or self.busy:
            return False
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Segmentation input must be uint8 RGB [H,W,3]')
        job_dir = self.directory / f'{self._index:06d}'
        job_dir.mkdir(parents=True, exist_ok=False)
        self._index += 1
        np.save(job_dir / 'rgb.npy', rgb, allow_pickle=False)
        request = dict(self.options, prompt=prompt, generation=generation,
                       source_id=source_id, timestamp_ns=timestamp_ns)
        (job_dir / 'request.json').write_text(json.dumps(request, indent=2, ensure_ascii=False))
        self._job = (generation, job_dir, rgb.shape[:2])
        self._cancelled_at = None
        self._log = (job_dir / 'process.log').open('w')
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-m', 'cordvip_geometry.segmentation_worker', '--job', str(job_dir.resolve())],
                env=module_environment(),
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
            )
        except Exception:
            self._log.close()
            self._log = None
            self._job = None
            raise
        return True

    def cancel(self):
        if self._process is not None and self._cancelled_at is None:
            self._cancelled_at = time.monotonic()
            if self._process.poll() is None:
                self._process.terminate()

    def poll(self):
        if self._process is None:
            return None
        if self._cancelled_at is not None and time.monotonic() - self._cancelled_at > 2:
            if self._process.poll() is None:
                self._process.kill()
        returncode = self._process.poll()
        if returncode is None:
            return None
        self._process.wait()
        self._process = None
        self._log.close()
        self._log = None
        generation, job_dir, shape = self._job
        self._job = None
        if self._cancelled_at is not None:
            self._cancelled_at = None
            return None
        candidates = []
        error = None
        try:
            if returncode:
                tail = (job_dir / 'process.log').read_text(errors='replace')[-1600:]
                raise RuntimeError(f'Grounded-SAM exited with {returncode}: {tail}')
            with np.load(job_dir / 'candidates.npz', allow_pickle=False) as data:
                masks, boxes = data['masks'], data['boxes']
                labels, scores, mask_scores = data['labels'], data['scores'], data['mask_scores']
                n = len(masks)
                if (masks.dtype != np.bool_ or masks.shape != (n, *shape)
                        or boxes.shape != (n, 4) or labels.shape != (n,)
                        or scores.shape != (n,) or mask_scores.shape != (n,)
                        or not all(np.isfinite(x).all() for x in (boxes, scores, mask_scores))):
                    raise ValueError('Invalid segmentation candidate shapes or values')
                candidates = [dict(mask=masks[i].copy(), box_xyxy=boxes[i].copy(),
                                   label=str(labels[i]), score=float(scores[i]),
                                   mask_score=float(mask_scores[i])) for i in range(n)]
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        return SegmentationResult(generation, candidates, job_dir, error)

    def close(self, timeout_s=0, *, force=False):
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError('Shutdown timeout must be finite and nonnegative')
        self._closed = True
        self.cancel()
        if force and self._process is not None and self._process.poll() is None:
            self._process.kill()
        if self._process is not None and timeout_s:
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
        self.poll()
        return CloseStatus(stopped=not self.busy, error=None)


def run_job(job_dir):
    # This entry point is invoked only after an explicit segmentation request.
    from .segmentation import GroundedSegmenter

    job_dir = Path(job_dir)
    request = json.loads((job_dir / 'request.json').read_text())
    rgb = np.load(job_dir / 'rgb.npy', allow_pickle=False)
    segmenter = GroundedSegmenter(
        dino_weights=request['dino_weights'], sam_root=request['sam_root'],
        sam_checkpoint=request['sam_checkpoint'], sam_config=request['sam_config'],
    )
    try:
        (job_dir / 'provenance.json').write_text(json.dumps(segmenter.provenance, indent=2, ensure_ascii=False))
        candidates = segmenter.predict(rgb, request['prompt'],
                                       box_threshold=request['box_threshold'],
                                       text_threshold=request['text_threshold'])
        masks = np.stack([c['mask'] for c in candidates]) if candidates else np.empty((0, *rgb.shape[:2]), bool)
        np.savez_compressed(
            job_dir / 'candidates.npz', masks=masks,
            boxes=np.asarray([c['box_xyxy'] for c in candidates], np.float32).reshape(-1, 4),
            labels=np.asarray([c['label'] for c in candidates], dtype=str),
            scores=np.asarray([c['score'] for c in candidates], np.float32),
            mask_scores=np.asarray([c['mask_score'] for c in candidates], np.float32),
        )
    finally:
        segmenter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    run_job(parser.parse_args(argv).job)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
