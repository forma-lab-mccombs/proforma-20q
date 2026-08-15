"""Fetch + md5-verify the gated ProForma-20Q data artifacts from Hugging Face.

The code in this repository is Apache-2.0. The *data* it consumes is not: the
regularization statistics, the coverage mask and its row index, and the
canonical per-column drift statistics are all derived from Compustat via WRDS,
whose licence is non-commercial. They are therefore distributed from gated
Hugging Face repositories under the Forma Non-Commercial Research Licence
rather than bundled into an Apache-2.0 tree -- the dataset bundle under the
WRDS-Conditioned variant, the regularization statistics (which ship with the
model weights) under the plain variant. See README.md ("Code and data are
licensed separately") and NOTICE.

Both gates are ``auto``: any Hugging Face account can accept the terms and get
immediate access. The dataset bundle's WRDS condition costs an acceptance
click, not a capability -- every user of those artifacts already holds a
WRDS/Compustat entitlement, since without one there is no panel to score
against. The regularization statistics carry no such condition.

Two repositories, deliberately:

* ``forma-lab-mccombs/forma`` (model) -- ``reference/regularization_stats.parquet``,
  co-located with the weights so the ``(mu, sigma, k)`` that define the target
  space have a single source of truth shared with the trained models.
* ``forma-lab-mccombs/proforma-20q-artifacts`` (dataset) -- the released
  forecasts, the coverage mask and its row index, and the canonical column
  statistics ``report-drift`` compares against.

Every fetch is md5-verified against a pin recorded in the calling module. A
truncated or substituted download fails loudly rather than scoring silently
against the wrong bytes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Gated repositories, by role. `url` is what a user without access must visit to
# accept the terms -- it is quoted verbatim in the error below, so a failure
# tells you exactly where to go rather than leaving you to guess the path from
# a repo id.
FORMA_MODEL_REPO = "forma-lab-mccombs/forma"
ARTIFACTS_REPO = "forma-lab-mccombs/proforma-20q-artifacts"

_REPO_URLS = {
    (FORMA_MODEL_REPO, "model"): f"https://huggingface.co/{FORMA_MODEL_REPO}",
    (ARTIFACTS_REPO, "dataset"): f"https://huggingface.co/datasets/{ARTIFACTS_REPO}",
}


def repo_url(repo_id: str, repo_type: str = "dataset") -> str:
    """The browser URL where a user accepts the gate for ``repo_id``."""
    if (repo_id, repo_type) in _REPO_URLS:
        return _REPO_URLS[(repo_id, repo_type)]
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


class GatedArtifactError(RuntimeError):
    """Raised when a gated artifact cannot be fetched.

    Covers every reason distinctly -- ``huggingface_hub`` not installed, no
    token, token without access, file absent from the repo -- because the fix
    differs for each and a generic "download failed" sends users to the wrong
    one.
    """


def md5sum(path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_md5(path, expected: str, label: str | None = None) -> None:
    """Raise unless ``path`` hashes to ``expected``.

    Callers stage downloads under a temporary name and only publish them at the
    real filename after this returns, so a failed verification can never leave a
    plausible-looking artifact where a later run would trust it.
    """
    label = label or str(path)
    got = md5sum(path)
    if got != expected:
        raise GatedArtifactError(
            f"md5 mismatch for {label}: got {got}, expected {expected}. "
            f"The download is truncated, substituted, or the pin is stale -- "
            f"refusing to use it (unverified file left at {path})."
        )


def _access_hint(repo_id: str, repo_type: str, detail: str) -> GatedArtifactError:
    url = repo_url(repo_id, repo_type)
    return GatedArtifactError(
        f"Cannot access the gated repository {repo_id} ({detail}).\n"
        f"\n"
        f"  1. Request access (accepted automatically) at:\n"
        f"       {url}\n"
        f"  2. Authenticate, either with:\n"
        f"       huggingface-cli login\n"
        f"     or by setting the HF_TOKEN environment variable to a token from\n"
        f"       https://huggingface.co/settings/tokens\n"
        f"\n"
        f"This artifact is derived from Compustat via WRDS and is licensed for\n"
        f"non-commercial research use; it is therefore not distributed inside\n"
        f"this Apache-2.0 repository. See README.md and NOTICE."
    )


def hf_download(repo_id: str, filename: str, *, repo_type: str = "dataset",
                local_dir=None, revision: str | None = None) -> Path:
    """``hf_hub_download`` with the failure modes turned into actionable errors.

    Returns the path to the downloaded file. Does **not** verify the md5 --
    callers pin their own digests and call :func:`verify_md5`, so the pin lives
    next to the artifact list it describes rather than in here.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise GatedArtifactError(
            "huggingface_hub is required to fetch the gated data artifacts.\n"
            "  pip install huggingface_hub\n"
            "(or `pip install -e .[data]` in a checkout of this repository)."
        ) from e

    # Import the error classes defensively: huggingface_hub has moved them
    # between `huggingface_hub.utils` and `huggingface_hub.errors` across
    # versions, and a missing name here would mask the real error.
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError, GatedRepoError, HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:  # pragma: no cover - older huggingface_hub layout
        from huggingface_hub.utils import (  # type: ignore[no-redef]
            EntryNotFoundError, GatedRepoError, HfHubHTTPError,
            RepositoryNotFoundError,
        )

    try:
        return Path(hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type,
            local_dir=local_dir, revision=revision,
        ))
    except GatedRepoError as e:
        raise _access_hint(repo_id, repo_type,
                           "your account has not been granted access") from e
    except RepositoryNotFoundError as e:
        # 404 is also what the Hub returns for a private/gated repo when the
        # token is absent or lacks read scope, so do not report it as "the repo
        # does not exist" -- that would send an unauthenticated user chasing a
        # typo instead of logging in.
        raise _access_hint(repo_id, repo_type,
                           "not found, or you are not authenticated") from e
    except EntryNotFoundError as e:
        raise GatedArtifactError(
            f"{repo_id} does not contain {filename!r}. The repository layout may "
            f"have changed; see {repo_url(repo_id, repo_type)}"
        ) from e
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise _access_hint(repo_id, repo_type,
                               f"HTTP {status}: not authenticated or not authorized") from e
        raise GatedArtifactError(
            f"Failed to download {filename!r} from {repo_id}: {e}") from e


def list_repo_files(repo_id: str, *, repo_type: str = "dataset",
                    revision: str | None = None) -> list[str]:
    """Every file path in a gated repository, with the same actionable errors.

    Used to resolve an artifact's remote path by basename instead of hard-coding
    a directory layout this repository does not control. Publishing the bundle
    is a separate job from consuming it, and a downloader that breaks when the
    publisher adds a subdirectory would be needlessly brittle.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise GatedArtifactError(
            "huggingface_hub is required to fetch the gated data artifacts.\n"
            "  pip install huggingface_hub"
        ) from e
    try:
        from huggingface_hub.errors import (
            GatedRepoError, HfHubHTTPError, RepositoryNotFoundError,
        )
    except ImportError:  # pragma: no cover - older huggingface_hub layout
        from huggingface_hub.utils import (  # type: ignore[no-redef]
            GatedRepoError, HfHubHTTPError, RepositoryNotFoundError,
        )
    try:
        return list(HfApi().list_repo_files(repo_id, repo_type=repo_type,
                                            revision=revision))
    except GatedRepoError as e:
        raise _access_hint(repo_id, repo_type,
                           "your account has not been granted access") from e
    except RepositoryNotFoundError as e:
        raise _access_hint(repo_id, repo_type,
                           "not found, or you are not authenticated") from e
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise _access_hint(repo_id, repo_type,
                               f"HTTP {status}: not authenticated or not authorized") from e
        raise GatedArtifactError(
            f"Failed to list {repo_id}: {e}") from e
    except OSError as e:
        # requests' ConnectionError/Timeout subclass OSError. This is the
        # offline case, and it must arrive as a GatedArtifactError like every
        # other failure here: callers catch that one class to print an
        # actionable message instead of a traceback.
        raise GatedArtifactError(
            f"Could not reach the Hugging Face Hub to list {repo_id}: {e}\n"
            f"Check your network connection (or HF_ENDPOINT, if you use a "
            f"mirror); see {repo_url(repo_id, repo_type)}") from e


def resolve_remote_path(name: str, listing, repo_id: str, repo_type: str = "dataset") -> str:
    """Find ``name`` in a repo ``listing``, by exact path or unique basename.

    Refuses an ambiguous basename rather than picking one: two files with the
    same name in different directories are not interchangeable, and silently
    choosing would md5-fail at best and score against the wrong bytes at worst.
    """
    if name in listing:
        return name
    matches = [p for p in listing if p.rsplit("/", 1)[-1] == name.rsplit("/", 1)[-1]]
    if len(matches) == 1:
        return matches[0]
    url = repo_url(repo_id, repo_type)
    if not matches:
        raise GatedArtifactError(
            f"{repo_id} does not contain {name!r} (searched {len(listing)} files). "
            f"It may not have been published yet -- see {url}")
    raise GatedArtifactError(
        f"{name!r} is ambiguous in {repo_id}: {', '.join(sorted(matches))}. "
        f"Pin an explicit path; see {url}")


def fetch_verified(repo_id: str, filename: str, expected_md5: str, *,
                   repo_type: str = "dataset", revision: str | None = None) -> Path:
    """Download ``filename`` into the local Hugging Face cache and verify its md5.

    Suits the small always-needed artifacts (the regularization statistics, the
    canonical column statistics), where the shared cache is the right home and
    the caller just wants a trustworthy path. Bulk downloads that belong in a
    user-chosen directory compose :func:`hf_download` and :func:`verify_md5`
    directly -- see ``scripts/download_artifacts.py``.
    """
    path = hf_download(repo_id, filename, repo_type=repo_type, revision=revision)
    verify_md5(path, expected_md5, label=f"{repo_id}:{filename}")
    return path
