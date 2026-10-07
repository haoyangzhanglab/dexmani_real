"""Publication races and failed decoder initialization, entirely offline."""

import errno
import json
from fractions import Fraction
from types import SimpleNamespace as NS

import pytest

from dexmani_real.recording import results
from dexmani_real.recording.storage import video
from dexmani_real.utils import atomic_io
from dexmani_real.utils.atomic_io import atomic_publish


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_concurrent_target_creation_never_overwrites(tmp_path, kind):
    source, target = tmp_path / "staging", tmp_path / "published"
    if kind == "directory":
        source.mkdir()
        (source / "payload").write_bytes(b"new")
    else:
        source.write_bytes(b"new")
    existing_inode = []

    def concurrent_publisher():
        # Deterministically create the target after preflight, before publication.
        if kind == "directory":
            target.mkdir()
        elif kind == "symlink":
            target.symlink_to(tmp_path / "missing")
        else:
            target.write_bytes(b"existing evidence")
        existing_inode.append(target.lstat().st_ino)
        return False

    with pytest.raises(FileExistsError):
        atomic_publish(source, target, cancelled=concurrent_publisher)
    assert source.exists()
    assert target.lstat().st_ino == existing_inode[0]
    if kind == "file":
        assert target.read_bytes() == b"existing evidence"
    elif kind == "directory":
        assert list(target.iterdir()) == []
    else:
        assert target.is_symlink() and not target.exists()


@pytest.mark.parametrize("directory", [False, True])
def test_publication_preserves_payload_and_moves_source(tmp_path, directory):
    source, target = tmp_path / "staging", tmp_path / "published"
    if directory:
        source.mkdir()
    payload = source / "payload" if directory else source
    payload.write_bytes(b"evidence")
    assert atomic_publish(source, target) == target
    assert not source.exists()
    assert (target / "payload" if directory else target).read_bytes() == b"evidence"


def test_cancelled_publication_retains_staging(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "published"
    source.write_bytes(b"evidence")
    with pytest.raises(RuntimeError, match="cancelled"):
        atomic_publish(source, target, cancelled=lambda: True)
    assert source.read_bytes() == b"evidence" and not target.exists()


@pytest.mark.parametrize("missing_symbol", [False, True])
def test_unsupported_atomic_rename_retains_staging(monkeypatch, tmp_path, missing_symbol):
    source, target = tmp_path / "staging", tmp_path / "published"
    source.write_bytes(b"evidence")

    def unsupported(*args):
        atomic_io.ctypes.set_errno(errno.EOPNOTSUPP)
        return -1

    library = NS() if missing_symbol else NS(renameat2=unsupported)
    monkeypatch.setattr(atomic_io.ctypes, "CDLL", lambda *a, **kw: library)
    with pytest.raises(OSError) as error:
        atomic_publish(source, target)
    assert error.value.errno == (errno.ENOSYS if missing_symbol else errno.EOPNOTSUPP)
    assert source.read_bytes() == b"evidence" and not target.exists()


def test_publication_rejects_embedded_null_without_truncating_path(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "published"
    source.write_bytes(b"evidence")
    with pytest.raises(ValueError, match="null"):
        atomic_publish(str(source) + "\0ignored", target)
    assert source.read_bytes() == b"evidence" and not target.exists()


def test_session_creation_cannot_overwrite_concurrent_result(monkeypatch, tmp_path):
    native_dump = results.atomic_json_dump

    def concurrent_result(obj, path, **kwargs):
        native_dump({"session_id": "other experiment"}, path)
        return native_dump(obj, path, **kwargs)

    monkeypatch.setattr(results, "atomic_json_dump", concurrent_result)
    with pytest.raises(FileExistsError):
        results.SessionResults(tmp_path, "policy")
    assert json.loads((tmp_path / "session_result.json").read_text()) == {
        "session_id": "other experiment"
    }
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_claimed_session_can_update_but_cannot_be_reopened(tmp_path):
    session = results.SessionResults(tmp_path, "policy")
    session.prepare(recording=False)
    session.entered(1)
    session.finish_attempt("operator")
    session.finish_session(outcome="finished", reason="quit", shutdown_clean=True)
    path = tmp_path / "session_result.json"
    saved = path.read_bytes()
    assert json.loads(saved)["state"] == "finished"
    with pytest.raises(FileExistsError):
        results.SessionResults(tmp_path, "policy")
    assert path.read_bytes() == saved


@pytest.mark.parametrize("close_fails_once", [False, True])
def test_decoder_releases_failed_open_before_retry(monkeypatch, tmp_path, close_fails_once):
    calls = []
    closes = []

    def close_failed():
        closes.append("failed")
        if close_fails_once and closes.count("failed") == 1:
            raise OSError("close failed")

    failed = NS(
        streams=NS(video=[NS(frames=2, average_rate=None)]),
        close=close_failed,
    )
    good = NS(
        streams=NS(video=[NS(frames=2, average_rate=30, time_base=Fraction(1, 30))]),
        decode=lambda stream: iter([NS(pts=0, time_base=Fraction(1, 30))]),
        close=lambda: closes.append("good"),
    )

    def open_container(*args):
        calls.append(1)
        return failed if len(calls) == 1 else good

    monkeypatch.setattr(video.av, "open", open_container)
    decoder = video.VideoDecoder(tmp_path / "synthetic.mp4")
    with pytest.raises(OSError if close_fails_once else ValueError):
        _ = decoder.frame_count
    assert closes == ["failed"]
    assert decoder.frame_count == 2
    assert closes.count("failed") == (2 if close_fails_once else 1)
    decoder.close()
    assert closes[-1] == "good"
