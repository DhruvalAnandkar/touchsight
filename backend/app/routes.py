"""HTTP routes for convert, job status, artifacts, and sample images."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from app.images import decode_image
from app.schemas import ConvertResponse, JobView, SampleView

router = APIRouter()


@router.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/convert", response_model=ConvertResponse)
async def convert(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    hint: str = Form(""),
) -> ConvertResponse:
    settings = request.app.state.settings
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="That image is larger than 8 MB.")
    try:
        decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    hint = " ".join(hint.split())[:240]
    job = request.app.state.jobs.create(data, hint)
    if job.status != "complete":
        background.add_task(request.app.state.jobs.execute, job.id, data, hint)
    return ConvertResponse(job_id=job.id)


@router.get("/api/jobs/{job_id}", response_model=JobView)
def job_status(job_id: str, request: Request) -> JobView:
    state = request.app.state.jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="No job with that id.")
    return request.app.state.jobs.view(state)


@router.get("/api/artifacts/{job_id}/{name}")
def artifact(job_id: str, name: str, request: Request) -> FileResponse:
    path = request.app.state.jobs.artifact_path(job_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail="That file is not ready.")
    if path.suffix == ".json":
        media_type = "application/json"
        filename = "touchsight-region-map.json"
    elif path.suffix == ".pdf":
        media_type = "application/pdf"
        filename = "touchsight-tactile.pdf"
    elif path.suffix == ".png":
        media_type = "image/png"
        filename = path.name
    elif path.suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
        filename = path.name
    elif path.suffix == ".stl":
        media_type = "model/stl"
        filename = "touchsight-relief.stl"
    elif path.name == "tactile.svg":
        media_type = "image/svg+xml"
        filename = "touchsight-tactile.svg"
    else:
        media_type = "image/svg+xml"
        filename = "touchsight-outline.svg"
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="inline",
    )


@router.get("/api/samples", response_model=list[SampleView])
def samples(request: Request) -> list[SampleView]:
    return request.app.state.jobs.samples()


@router.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: str, request: Request) -> Response:
    path = request.app.state.jobs.sample_file(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown sample.")
    return FileResponse(path, media_type="image/png")
