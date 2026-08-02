"""Resumable concurrent image downloader.

Fakeddit ships URLs, not pixels, and a meaningful share of those Reddit links
have rotted since 2019. So this fetcher is built to be re-run: it skips what is
already on disk, records permanent failures so they are not retried forever, and
verifies that what came back actually decodes as an image.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from mcm.utils.logging import get_logger

log = get_logger(__name__)

# Guards against pathological responses from an uncurated URL list.
MAX_BYTES = 12 * 1024 * 1024
TIMEOUT = 15
USER_AGENT = "mcm-research-crawler/0.1 (academic dataset reconstruction)"
MIN_DIMENSION = 32


@dataclass
class FetchResult:
    key: str
    path: str | None
    ok: bool
    error: str = ""


def _dest_for(out_dir: Path, key: str) -> Path:
    """Shard into 256 subdirs — a single directory with 40k+ files is slow on APFS."""
    shard = hashlib.md5(key.encode()).hexdigest()[:2]
    d = out_dir / shard
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.jpg"


def _fetch_one(key: str, url: str, out_dir: Path, session: requests.Session) -> FetchResult:
    dest = _dest_for(out_dir, key)
    if dest.exists() and dest.stat().st_size > 0:
        return FetchResult(key, str(dest), True)

    try:
        resp = session.get(url, timeout=TIMEOUT, stream=True)
        if resp.status_code != 200:
            return FetchResult(key, None, False, f"http_{resp.status_code}")

        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return FetchResult(key, None, False, f"content_type_{ctype[:32]}")

        buf = BytesIO()
        for chunk in resp.iter_content(64 * 1024):
            buf.write(chunk)
            if buf.tell() > MAX_BYTES:
                return FetchResult(key, None, False, "too_large")

        # Decode before writing: Reddit serves HTML error pages and 1x1 tracking
        # pixels with image content-types, and both would poison training.
        buf.seek(0)
        img = Image.open(buf)
        img.verify()
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if min(img.size) < MIN_DIMENSION:
            return FetchResult(key, None, False, f"too_small_{img.size}")

        img.save(dest, format="JPEG", quality=92)
        return FetchResult(key, str(dest), True)

    except Exception as e:  # noqa: BLE001 - any failure is just a dead URL
        return FetchResult(key, None, False, f"{type(e).__name__}")


def fetch_images(
    items: list[tuple[str, str]],
    out_dir: Path,
    max_workers: int = 16,
    failure_log: Path | None = None,
) -> dict[str, str]:
    """Download ``(key, url)`` pairs into ``out_dir``.

    Returns a mapping of key -> absolute path for successes only. Callers are
    expected to drop rows whose key is absent rather than train on a missing
    image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_log = failure_log or out_dir.parent / "fetch_failures.json"

    # Previously-failed URLs are dead URLs; re-requesting them every run wastes
    # minutes and hammers the host for nothing.
    known_bad: dict[str, str] = {}
    if failure_log.exists():
        known_bad = json.loads(failure_log.read_text())
        items = [(k, u) for k, u in items if k not in known_bad]
        log.info("skipping %d URLs that failed on a previous run", len(known_bad))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: dict[str, str] = {}
    failures: dict[str, str] = dict(known_bad)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, k, u, out_dir, session): k for k, u in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images", unit="img"):
            r = fut.result()
            if r.ok and r.path:
                results[r.key] = r.path
            else:
                failures[r.key] = r.error

    failure_log.write_text(json.dumps(failures, indent=2))
    n_new = len(failures) - len(known_bad)
    log.info(
        "fetched %d/%d images (%d new failures, %d total recorded)",
        len(results),
        len(items),
        n_new,
        len(failures),
    )
    return results


def index_existing(out_dir: Path) -> dict[str, str]:
    """Map key -> path for images already on disk, without touching the network."""
    if not out_dir.exists():
        return {}
    return {p.stem: str(p) for p in out_dir.rglob("*.jpg")}
