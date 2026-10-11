"""In-memory jobs with a hash cache on disk.

A repeated upload of the same bytes returns a completed job immediately.
Precomputed hero images live in `demo_cache` and are checked before the
runtime cache, so the demo path does not depend on a live model.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pipeline.providers.sam2 import ReplicateSam2, Sam2Client
from pipeline.understand import cache_key, sample_hint
from pipeline.run import run_outline
from pipeline.types import STAGE_ORDER, StageUpdate

from app.config import Settings
from app.images import decode_image, image_hash
from app.schemas import CacheMeta, JobState, JobView, SampleInfo, SampleView, StageProgress

logger = logging.getLogger("touchsight.jobs")

OUTLINE_NAME = "outline.svg"
TACTILE_SVG_NAME = "tactile.svg"
TACTILE_PDF_NAME = "tactile.pdf"
REGION_MAP_NAME = "region_map.json"
RELIEF_STL_NAME = "relief.stl"
RECTIFY_BEFORE_NAME = "rectify-before.jpg"
RECTIFY_AFTER_NAME = "rectify-after.jpg"
ARTIFACT_NAMES = (
    OUTLINE_NAME,
    TACTILE_SVG_NAME,
    TACTILE_PDF_NAME,
    REGION_MAP_NAME,
    RELIEF_STL_NAME,
    RECTIFY_BEFORE_NAME,
    RECTIFY_AFTER_NAME,
)


class JobStore:
    def __init__(self, settings: Settings, sam: Sam2Client | None = None) -> None:
        self.settings = settings
        self.sam = sam
        self.jobs_dir = settings.data_dir / "jobs"
        self.cache_dir = settings.data_dir / "cache"
        self.demo_dir = settings.demo_cache_dir / "by_hash"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, data: bytes, hint: str = "") -> JobState:
        digest = image_hash(data)
        job_id = uuid.uuid4().hex
        cached = self._read_cache(digest, hint)
        if cached is not None:
            state = JobState(
                id=job_id,
                status="complete",
                image_hash=digest,
                cached=True,
                stages=cached.stages,
                region_count=cached.region_count,
                source=cached.source,
                created_at=_now(),
            )
            self._write_job(state)
            job_dir = self._job_dir(job_id)
            for name in ARTIFACT_NAMES:
                source = self._cached_file(digest, hint, name)
                if source is not None:
                    shutil.copyfile(source, job_dir / name)
            return state
        state = JobState(
            id=job_id,
            status="queued",
            image_hash=digest,
            stages=[StageProgress(name=name, status="pending") for name in STAGE_ORDER],
            created_at=_now(),
        )
        self._write_job(state)
        return state

    def execute(self, job_id: str, data: bytes, hint: str = "") -> None:
        self._update(job_id, status="running")
        try:
            image = decode_image(data)

            def on_progress(update: StageUpdate) -> None:
                self._apply_stage(job_id, update)

            result = run_outline(
                image,
                self.settings.tactile(),
                max_regions=self.settings.max_regions,
                sam=self.sam,
                image_hash=image_hash(data),
                vlm=self.settings.vlm(),
                hint=hint,
                image_bytes=data,
                on_progress=on_progress,
            )
        except Exception:
            logger.exception("job %s failed", job_id)
            self._fail(job_id, "Could not build a tactile outline from that image.")
            return
        digest = image_hash(data)
        job_dir = self._job_dir(job_id)
        (job_dir / OUTLINE_NAME).write_text(result.svg, encoding="utf-8")
        (job_dir / TACTILE_SVG_NAME).write_text(result.svg, encoding="utf-8")
        if result.pdf:
            (job_dir / TACTILE_PDF_NAME).write_bytes(result.pdf)
        if result.stl:
            (job_dir / RELIEF_STL_NAME).write_bytes(result.stl)
        if result.rectify_before:
            (job_dir / RECTIFY_BEFORE_NAME).write_bytes(result.rectify_before)
        if result.rectify_after:
            (job_dir / RECTIFY_AFTER_NAME).write_bytes(result.rectify_after)
        region_map = json.dumps(result.region_map(), indent=2)
        (job_dir / REGION_MAP_NAME).write_text(region_map, encoding="utf-8")
        state = self.get(job_id)
        if state is None:
            return
        meta = CacheMeta(
            image_hash=digest,
            region_count=len(result.regions),
            source=result.source,
            stages=state.stages,
        )
        self._write_cache(
            cache_key(digest, hint),
            result.svg,
            meta,
            region_map,
            result.pdf,
            result.stl,
            result.rectify_before,
            result.rectify_after,
        )
        self._update(
            job_id,
            status="complete",
            region_count=len(result.regions),
            source=result.source,
        )

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
        if state is not None:
            return state
        path = self._job_dir(job_id) / "job.json"
        if not path.is_file():
            return None
        state = JobState.model_validate_json(path.read_text(encoding="utf-8"))
        with self._lock:
            self._jobs[job_id] = state
        return state

    def view(self, state: JobState) -> JobView:
        artifacts: dict[str, str] = {}
        job_dir = self._job_dir(state.id)
        if (job_dir / OUTLINE_NAME).is_file():
            artifacts["outline_svg"] = f"/api/artifacts/{state.id}/{OUTLINE_NAME}"
        if (job_dir / TACTILE_SVG_NAME).is_file():
            artifacts["tactile_svg"] = f"/api/artifacts/{state.id}/{TACTILE_SVG_NAME}"
        if (job_dir / TACTILE_PDF_NAME).is_file():
            artifacts["tactile_pdf"] = f"/api/artifacts/{state.id}/{TACTILE_PDF_NAME}"
        if (job_dir / REGION_MAP_NAME).is_file():
            artifacts["region_map"] = f"/api/artifacts/{state.id}/{REGION_MAP_NAME}"
        if (job_dir / RELIEF_STL_NAME).is_file():
            artifacts["relief_stl"] = f"/api/artifacts/{state.id}/{RELIEF_STL_NAME}"
        if (job_dir / RECTIFY_BEFORE_NAME).is_file():
            artifacts["rectify_before"] = f"/api/artifacts/{state.id}/{RECTIFY_BEFORE_NAME}"
        if (job_dir / RECTIFY_AFTER_NAME).is_file():
            artifacts["rectify_after"] = f"/api/artifacts/{state.id}/{RECTIFY_AFTER_NAME}"
        return JobView(**state.model_dump(), artifacts=artifacts)

    def artifact_path(self, job_id: str, name: str) -> Path | None:
        if name not in ARTIFACT_NAMES:
            return None
        path = (self._job_dir(job_id) / name).resolve()
        root = self.jobs_dir.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        return path

    def samples(self) -> list[SampleView]:
        views: list[SampleView] = []
        for sample in self._manifest():
            views.append(
                SampleView(
                    id=sample.id,
                    title=sample.title,
                    description=sample.description,
                    hash=sample.hash,
                    image_url=f"/api/samples/{sample.id}/image",
                )
            )
        return views

    def sample_file(self, sample_id: str) -> Path | None:
        for sample in self._manifest():
            if sample.id != sample_id:
                continue
            root = self.settings.demo_cache_dir.resolve()
            path = (self.settings.demo_cache_dir / sample.file).resolve()
            if path.is_file() and path.is_relative_to(root):
                return path
        return None

    def _apply_stage(self, job_id: str, update: StageUpdate) -> None:
        def mutate(state: JobState) -> None:
            for index, stage in enumerate(state.stages):
                if stage.name != update.name:
                    continue
                state.stages[index] = StageProgress(
                    name=update.name,
                    status=update.status,
                    elapsed_ms=update.elapsed_ms,
                    message=update.message,
                    provider=update.provider,
                )
                return

        self._update(job_id, mutate=mutate)

    def _fail(self, job_id: str, message: str) -> None:
        def mutate(state: JobState) -> None:
            state.status = "failed"
            state.error = message
            for index, stage in enumerate(state.stages):
                if stage.status == "running":
                    state.stages[index] = stage.model_copy(update={"status": "failed", "message": message})

        self._update(job_id, mutate=mutate)

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        region_count: int | None = None,
        source: str | None = None,
        mutate: object | None = None,
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            if status is not None:
                state.status = status
            if region_count is not None:
                state.region_count = region_count
            if source is not None:
                state.source = source
            if callable(mutate):
                mutate(state)
            self._persist(state)

    def _write_job(self, state: JobState) -> None:
        with self._lock:
            self._jobs[state.id] = state
            self._persist(state)

    def _persist(self, state: JobState) -> None:
        directory = self._job_dir(state.id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "job.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def _job_dir(self, job_id: str) -> Path:
        safe = "".join(char for char in job_id if char.isalnum())
        return self.jobs_dir / safe

    def _cache_dirs(self, digest: str, hint: str) -> list[Path]:
        cleaned = " ".join(hint.split())
        directories: list[Path] = []
        if not cleaned or cleaned == sample_hint(digest):
            directories.append(self.demo_dir / digest)
        directories.append(self.cache_dir / cache_key(digest, hint))
        return directories

    def _read_cache(self, digest: str, hint: str = "") -> CacheMeta | None:
        for directory in self._cache_dirs(digest, hint):
            meta_path = directory / "meta.json"
            svg_path = directory / OUTLINE_NAME
            if not meta_path.is_file() or not svg_path.is_file():
                continue
            meta = CacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if self._skip_cached_labels(meta):
                continue
            return meta
        return None

    def _svg_path(self, digest: str) -> Path:
        for directory in (self.demo_dir, self.cache_dir):
            svg_path = directory / digest / OUTLINE_NAME
            if svg_path.is_file():
                return svg_path
        raise FileNotFoundError(digest)

    def _cached_file(self, digest: str, hint: str, name: str) -> Path | None:
        for directory in self._cache_dirs(digest, hint):
            path = directory / name
            if path.is_file():
                return path
        return None

    def _skip_cached_labels(self, meta: CacheMeta) -> bool:
        """A live vision key should try again instead of keeping a fallback forever."""

        provider = ""
        for stage in meta.stages:
            if stage.name == "understand":
                provider = stage.provider or ""
                break
        if provider not in {"fallback", "sample-labels"}:
            return False
        vlm = self.settings.vlm()
        return bool(vlm.provider.strip() and vlm.api_key.strip())

    def _write_cache(
        self,
        digest: str,
        svg: str,
        meta: CacheMeta,
        region_map: str,
        pdf: bytes,
        stl: bytes,
        result_before: bytes = b"",
        result_after: bytes = b"",
    ) -> None:
        if self._skip_cached_labels(meta):
            return
        directory = self.cache_dir / digest
        directory.mkdir(parents=True, exist_ok=True)
        (directory / OUTLINE_NAME).write_text(svg, encoding="utf-8")
        (directory / TACTILE_SVG_NAME).write_text(svg, encoding="utf-8")
        if pdf:
            (directory / TACTILE_PDF_NAME).write_bytes(pdf)
        if stl:
            (directory / RELIEF_STL_NAME).write_bytes(stl)
        if result_before:
            (directory / RECTIFY_BEFORE_NAME).write_bytes(result_before)
        if result_after:
            (directory / RECTIFY_AFTER_NAME).write_bytes(result_after)
        (directory / REGION_MAP_NAME).write_text(region_map, encoding="utf-8")
        (directory / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def _manifest(self) -> list[SampleInfo]:
        path = self.settings.demo_cache_dir / "manifest.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [SampleInfo.model_validate(item) for item in payload.get("samples", [])]


def build_sam(settings: Settings) -> Sam2Client | None:
    token = settings.sam_api_token.strip()
    if not token:
        return None
    return ReplicateSam2(
        token,
        model=settings.sam_model,
        api_base=settings.sam_api_base,
        timeout_s=settings.sam_timeout_s,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
