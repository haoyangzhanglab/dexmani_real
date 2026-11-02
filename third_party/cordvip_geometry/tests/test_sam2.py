import io
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cordvip_geometry import sam2
from cordvip_geometry.models import SAM_MODELS, resolve_sam_model
from cordvip_geometry.sam2 import SAM2Client


def test_sam_resolves_official_and_custom_pairs(tmp_path):
    checkpoint, config = resolve_sam_model(tmp_path)
    assert checkpoint == tmp_path / SAM_MODELS["small"]["weight"]
    assert config == SAM_MODELS["small"]["config"]
    checkpoint, config = resolve_sam_model(tmp_path, variant="large")
    assert checkpoint == tmp_path / SAM_MODELS["large"]["weight"]
    assert config == SAM_MODELS["large"]["config"]
    custom = tmp_path / "custom.pt"
    assert resolve_sam_model(tmp_path, checkpoint=custom, config="custom.yaml") == (
        custom,
        "custom.yaml",
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"checkpoint": Path("custom.pt")},
        {"config": "custom.yaml"},
        {"variant": "small", "checkpoint": Path("custom.pt"), "config": "custom.yaml"},
    ],
)
def test_sam_rejects_incomplete_or_conflicting_pairs(tmp_path, parameters):
    with pytest.raises(ValueError):
        resolve_sam_model(tmp_path, **parameters)


def make_client(tmp_path):
    return SAM2Client(
        "/unused/python",
        tmp_path,
        tmp_path,
        checkpoint=tmp_path / "unused.pt",
        config="unused.yaml",
        log_path=tmp_path / "worker.log",
    )


def test_context_closes_log_when_process_cannot_start(tmp_path):
    client = make_client(tmp_path)
    with (
        patch.object(
            sam2.subprocess, "Popen", side_effect=FileNotFoundError("python")
        ) as spawn,
        pytest.raises(FileNotFoundError),
        client,
    ):
        pytest.fail("startup should have failed")
    assert spawn.call_args.kwargs["stderr"].closed
    assert client._process is None and client._log is None


@pytest.mark.parametrize("reply", [None, {"error": "model load failed"}])
def test_context_closes_process_pipes_and_log_when_worker_startup_fails(
    tmp_path, reply
):
    client = make_client(tmp_path)
    output = io.BytesIO()
    if reply is not None:
        pickle.dump((reply, {}), output, protocol=5)
        output.seek(0)
    process = MagicMock(stdin=io.BytesIO(), stdout=output, returncode=None)
    process.wait.side_effect = lambda **kwargs: setattr(process, "returncode", 1)
    with (
        patch.object(sam2.subprocess, "Popen", return_value=process) as spawn,
        patch.object(sam2.os, "killpg"),
        pytest.raises(RuntimeError, match="log:"),
        client,
    ):
        pytest.fail("startup should have failed")
    assert process.stdin.closed and process.stdout.closed
    assert spawn.call_args.kwargs["stderr"].closed
    assert client._process is None and client._log is None


def test_context_preserves_primary_error_when_cleanup_also_fails(tmp_path):
    client = make_client(tmp_path)
    with (
        patch.object(client, "start", return_value=client),
        patch.object(client, "close", side_effect=OSError("cleanup")),
        pytest.raises(ValueError, match="primary"),
        client,
    ):
        raise ValueError("primary")


def test_context_reports_cleanup_error_after_success(tmp_path):
    client = make_client(tmp_path)
    with (
        patch.object(client, "start", return_value=client),
        patch.object(client, "close", side_effect=OSError("cleanup")),
        pytest.raises(OSError, match="cleanup"),
        client,
    ):
        pass


def test_context_reports_truncated_worker_reply_and_closes_resources(tmp_path):
    client = make_client(tmp_path)
    output = io.BytesIO(pickle.dumps(({"provenance": {}}, {}), protocol=5)[:-1])
    process = MagicMock(stdin=io.BytesIO(), stdout=output, returncode=None)
    process.wait.side_effect = lambda **kwargs: setattr(process, "returncode", 1)
    with (
        patch.object(sam2.subprocess, "Popen", return_value=process),
        patch.object(sam2.os, "killpg"),
        pytest.raises(RuntimeError, match="worker stopped") as failure,
        client,
    ):
        pytest.fail("startup should have failed")
    assert isinstance(failure.value.__cause__, pickle.UnpicklingError)
    assert process.stdin.closed and process.stdout.closed
    assert client._process is None and client._log is None


def test_default_log_inherits_stderr_without_creating_files(tmp_path):
    client = SAM2Client(
        "/unused/python",
        tmp_path,
        tmp_path,
        checkpoint=tmp_path / "unused.pt",
        config="unused.yaml",
    )
    output = io.BytesIO()
    pickle.dump(({"provenance": {}}, {}), output, protocol=5)
    output.seek(0)
    process = MagicMock(stdin=io.BytesIO(), stdout=output, returncode=None)
    process.wait.side_effect = lambda **kwargs: setattr(process, "returncode", 0)
    with (
        patch.object(sam2.subprocess, "Popen", return_value=process) as spawn,
        patch.object(sam2.os, "killpg"),
        client,
    ):
        assert client.log_path is None and client._log is None
        assert spawn.call_args.kwargs["stderr"] is None
    assert process.stdin.closed and process.stdout.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["startup", "inference", "eof"])
def test_default_log_preserves_worker_errors_and_closes_resources(tmp_path, failure):
    client = SAM2Client(
        "/unused/python",
        tmp_path,
        tmp_path,
        checkpoint=tmp_path / "unused.pt",
        config="unused.yaml",
    )
    output = io.BytesIO()
    if failure != "startup":
        pickle.dump(({"provenance": {}}, {}), output, protocol=5)
    if failure != "eof":
        pickle.dump(({"error": f"{failure} failed"}, {}), output, protocol=5)
    output.seek(0)
    process = MagicMock(stdin=io.BytesIO(), stdout=output, returncode=None)
    process.wait.side_effect = lambda **kwargs: setattr(process, "returncode", 1)
    expected = "worker stopped" if failure == "eof" else f"{failure} failed"
    with (
        patch.object(sam2.subprocess, "Popen", return_value=process) as spawn,
        patch.object(sam2.os, "killpg"),
        pytest.raises(RuntimeError, match=f"{expected}; stderr"),
        client,
    ):
        client.predict(np.zeros((3, 4, 3), dtype=np.uint8), "object.")
    assert spawn.call_args.kwargs["stderr"] is None
    assert process.stdin.closed and process.stdout.closed
    assert client._process is None and client._log is None
    assert not list(tmp_path.iterdir())


FAKE_MODEL = """
import numpy as np
from pathlib import Path
import cordvip_geometry.sam2_worker as worker

class Model:
    def __init__(self, root, checkpoint, config, dino_weights):
        if Path(checkpoint).name == 'fail.pt':
            raise RuntimeError('fixture startup failure')
        print('fixture model loaded', flush=True)
        self.provenance = {'fixture': True}

    def predict(self, rgb, text, box_threshold, text_threshold):
        if text == 'fail.':
            raise RuntimeError('fixture inference failure')
        count = 0 if text == 'empty.' else 1
        return {
            'masks': (rgb[..., 0] > 100)[None][:count],
            'boxes': np.zeros((count, 4), dtype=np.float32),
            'scores': np.ones(count, dtype=np.float32),
            'mask_scores': np.ones(count, dtype=np.float32),
        }, ['object'][:count]

    def initialize(self, rgb, ids, masks):
        self.masks = masks.copy()
        return self.masks

    def track(self, rgb):
        return self.masks

worker.GroundedSAM2 = Model
raise SystemExit(worker.main())
"""


def worker_client(tmp_path, *, checkpoint="unused.pt"):
    return SAM2Client(
        sys.executable,
        tmp_path,
        tmp_path,
        checkpoint=tmp_path / checkpoint,
        config="unused.yaml",
        log_path=tmp_path / "worker.log",
    )


def fake_model_process(command, **kwargs):
    return REAL_POPEN([sys.executable, "-u", "-c", FAKE_MODEL, *command[4:]], **kwargs)


REAL_POPEN = sam2.subprocess.Popen


def test_private_worker_transfers_noncontiguous_rgb_and_boolean_masks(tmp_path):
    client = worker_client(tmp_path)
    rgb = np.arange(6 * 16 * 3, dtype=np.uint8).reshape(6, 16, 3)[:, ::2]
    assert not rgb.flags.c_contiguous
    with patch.object(sam2.subprocess, "Popen", side_effect=fake_model_process), client:
        candidate = client.predict(rgb, "object.")[0]
        assert candidate["mask"].dtype == np.bool_
        assert candidate["mask"].shape == rgb.shape[:2]
        assert candidate["box_xyxy"].dtype == np.float32
        np.testing.assert_array_equal(candidate["mask"], rgb[..., 0] > 100)
        assert client.predict(rgb, "empty.") == []
        masks = {"object": candidate["mask"]}
        for result in (client.initialize(rgb, masks), client.track(rgb)):
            assert list(result) == ["object"]
            assert result["object"].dtype == np.bool_
            np.testing.assert_array_equal(result["object"], masks["object"])
    assert client._process is None and client._log is None
    assert "fixture model loaded" in (tmp_path / "worker.log").read_text()


@pytest.mark.parametrize("failure", ["startup", "inference"])
def test_private_worker_reports_model_failure_and_closes_resources(tmp_path, failure):
    client = worker_client(
        tmp_path, checkpoint="fail.pt" if failure == "startup" else "unused.pt"
    )
    with (
        patch.object(sam2.subprocess, "Popen", side_effect=fake_model_process),
        pytest.raises(RuntimeError, match=f"fixture {failure} failure"),
        client,
    ):
        client.predict(np.zeros((3, 4, 3), dtype=np.uint8), "fail.")
    assert client._process is None and client._log is None
    log = (tmp_path / "worker.log").read_text()
    assert "Traceback" in log and f"fixture {failure} failure" in log
