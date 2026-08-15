"""Tests for the gated-artifact plumbing (``proforma20q.gated``).

Hermetic: every Hugging Face entry point is stubbed. What is under test is the
*error taxonomy* -- a failure has to arrive as a ``GatedArtifactError`` carrying
the fix, because that is the one class the callers catch. Anything else reaches
the user as a traceback and loses the message the module exists to print.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from proforma20q.gated import (
    ARTIFACTS_REPO, GatedArtifactError, list_repo_files, repo_url,
)

_ROOT = Path(__file__).resolve().parent.parent


class _Resp:
    """Just enough of a ``requests.Response`` for HfHubHTTPError to be built."""

    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.text = ""
        self.url = "https://huggingface.co/(test)"
        self.request = None


def _http_error(errors, message, status_code):
    """``HfHubHTTPError`` across versions: newer releases require ``response=``."""
    response = _Resp(status_code)
    try:
        return errors.HfHubHTTPError(message, response=response)
    except TypeError:  # pragma: no cover - older huggingface_hub signature
        exc = errors.HfHubHTTPError(message)
        exc.response = response
        return exc


def _stub_hfapi(monkeypatch, exc):
    hub = pytest.importorskip("huggingface_hub")

    class _Api:
        def list_repo_files(self, *a, **k):
            raise exc

    monkeypatch.setattr(hub, "HfApi", lambda *a, **k: _Api())


def test_list_repo_files_turns_an_offline_failure_into_a_gated_error(monkeypatch):
    """Offline, ``HfApi().list_repo_files`` raises a bare connection error. It is
    the first network call ``scripts/download_artifacts.py`` makes, and that
    script catches ``GatedArtifactError`` only -- so anything else exits with a
    raw traceback instead of the actionable message."""
    _stub_hfapi(monkeypatch, ConnectionError("name resolution failed (test)"))

    with pytest.raises(GatedArtifactError) as e:
        list_repo_files(ARTIFACTS_REPO, repo_type="dataset")
    assert "Could not reach the Hugging Face Hub" in str(e.value)
    assert ARTIFACTS_REPO in str(e.value)


def test_list_repo_files_turns_http_403_into_the_access_hint(monkeypatch):
    """A 403 on the listing is the unaccepted-gate case, and must produce the
    same 'accept the terms here, authenticate like this' hint as a download."""
    errors = pytest.importorskip("huggingface_hub.errors")

    exc = _http_error(errors, "forbidden (test)", 403)
    _stub_hfapi(monkeypatch, exc)

    with pytest.raises(GatedArtifactError) as e:
        list_repo_files(ARTIFACTS_REPO, repo_type="dataset")
    msg = str(e.value)
    assert repo_url(ARTIFACTS_REPO, "dataset") in msg
    assert "HF_TOKEN" in msg


def test_list_repo_files_turns_a_server_error_into_a_gated_error(monkeypatch):
    """A 500 is not an access problem, so it must not send the user to the gate
    page -- but it still has to arrive as GatedArtifactError."""
    errors = pytest.importorskip("huggingface_hub.errors")

    exc = _http_error(errors, "internal server error (test)", 500)
    _stub_hfapi(monkeypatch, exc)

    with pytest.raises(GatedArtifactError, match="Failed to list"):
        list_repo_files(ARTIFACTS_REPO, repo_type="dataset")


def test_credential_free_path_never_imports_huggingface_hub():
    """``validate`` and ``evaluate --truth`` are the documented no-login path: no
    WRDS account, no Hugging Face token, no network. Asserting the module is
    never even imported is what makes that testable -- an import is the only way
    a fetch could happen. Runs in a subprocess because another test in this file
    imports huggingface_hub into this interpreter."""
    code = (
        "import sys\n"
        "from proforma20q.cli import main\n"
        "rc = main(['validate', 'examples/example_forecast.parquet'])\n"
        "rc += main(['evaluate', 'examples/example_forecast.parquet',\n"
        "            '--truth', 'examples/example_truth.parquet'])\n"
        "hf = sorted(m for m in sys.modules if m.split('.')[0] == 'huggingface_hub')\n"
        "assert not hf, hf\n"
        "sys.exit(rc)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT), env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
