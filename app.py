from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from inference import FlowerPredictor, ModelNotReady

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "bloomwatch.db"
STATIC_DIR = BASE_DIR / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
CONFIDENCE_THRESHOLD = float(os.getenv("BLOOMWATCH_CONFIDENCE_THRESHOLD", "0.60"))

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
predictor = FlowerPredictor(BASE_DIR / "model")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                predicted_species TEXT,
                confidence REAL,
                top_predictions TEXT NOT NULL DEFAULT '[]',
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                observed_at TEXT NOT NULL,
                observer TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT NOT NULL DEFAULT 'field',
                image_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_species ON sightings(species)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON sightings(status)")


def seed_sample_data() -> None:
    if os.getenv("BLOOMWATCH_SEED_SAMPLES", "1") != "1":
        return
    samples = [
        ("Rose", 0.94, 26.8467, 80.9462, 1, "verified", "Campus garden transect"),
        ("Sunflower", 0.88, 26.8601, 80.9270, 2, "pending", "Open patch near field station"),
        ("Hibiscus", 0.91, 26.8384, 80.9531, 4, "verified", "Roadside garden"),
        ("Marigold", 0.83, 26.8508, 80.9654, 7, "verified", "Community plantation"),
        ("Lotus", 0.57, 26.8312, 80.9187, 9, "pending", "Pond edge; low-confidence example"),
        ("Rose", 0.89, 26.8570, 80.9490, 12, "verified", "Second recorded sighting"),
    ]
    now = datetime.now(timezone.utc)
    with connect() as connection:
        if connection.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]:
            return
        connection.executemany(
            """
            INSERT INTO sightings
            (species, predicted_species, confidence, top_predictions, latitude, longitude,
             observed_at, observer, notes, status, source, image_path, created_at)
            VALUES (?, ?, ?, '[]', ?, ?, ?, 'Sample survey', ?, ?, 'sample', NULL, ?)
            """,
            [
                (
                    species,
                    species,
                    confidence,
                    latitude,
                    longitude,
                    (now - timedelta(days=days)).isoformat(),
                    notes,
                    status,
                    now.isoformat(),
                )
                for species, confidence, latitude, longitude, days, status, notes in samples
            ],
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    seed_sample_data()
    yield


app = FastAPI(title="BloomWatch", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


class SightingUpdate(BaseModel):
    status: Literal["pending", "verified", "rejected"] | None = None
    species: str | None = None


def row_dict(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["image_url"] = f"/uploads/{result['image_path']}" if result.get("image_path") else None
    result.pop("image_path", None)
    try:
        result["top_predictions"] = json.loads(result.get("top_predictions") or "[]")
    except json.JSONDecodeError:
        result["top_predictions"] = []
    return result


async def read_valid_image(upload: UploadFile) -> Image.Image:
    content = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 10 MB.")
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Upload a valid JPG, PNG, or WebP image.") from exc


def parse_observed_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid observation date/time.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def model_status() -> dict:
    model_name = predictor.model_path.name if predictor.model_path else "checkpoint"
    messages = {
        "model": f"Connected to {model_name} ({predictor.framework}) with {len(predictor.class_names)} classes.",
        "preview": "Interface preview: identification results and initial sightings are illustrative, not model output.",
        "error": "A checkpoint was found but could not be loaded. See the model setup message.",
    }
    return {
        "mode": predictor.mode,
        "model_loaded": predictor.ready,
        "model_file": predictor.model_path.name if predictor.model_path else None,
        "class_count": predictor.class_count,
        "labels_loaded": predictor.labels_loaded,
        "device": predictor.device,
        "framework": predictor.framework,
        "preprocessing": predictor.config.get("preprocessing", "resnet50") if predictor.framework == "keras" else None,
        "input_size": list(predictor._input_size) if predictor.framework == "keras" else None,
        "error": predictor.error,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "message": messages[predictor.mode],
    }


@app.get("/api/classes")
def classes() -> dict:
    return {"classes": predictor.class_names, "count": len(predictor.class_names)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image = await read_valid_image(file)
    demo = not predictor.ready
    if demo:
        # Explicitly illustrative UI data. No claim is made that these values derive from the image.
        predictions = [
            {"species": "Rose", "class_index": -1, "confidence": 0.91},
            {"species": "Hibiscus", "class_index": -1, "confidence": 0.06},
            {"species": "Marigold", "class_index": -1, "confidence": 0.03},
        ]
        notice = "Illustrative preview only — this result was not generated from your image. Connect your checkpoint for real inference."
    else:
        try:
            predictions = predictor.predict(image, top_k=3)
            notice = "Model scores are decision support, not certainty. Verify low-confidence or unusual records."
        except ModelNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return {
        "predictions": predictions,
        "needs_review": predictions[0]["confidence"] < CONFIDENCE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "demo": demo,
        "notice": notice,
    }


@app.get("/api/sightings")
def list_sightings(
    species: str = "",
    status: str = "",
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict:
    clauses: list[str] = []
    parameters: list[object] = []
    if species:
        clauses.append("species = ?")
        parameters.append(species)
    if status:
        clauses.append("status = ?")
        parameters.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    parameters.append(limit)
    with connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM sightings {where} ORDER BY observed_at DESC LIMIT ?", parameters
        ).fetchall()
    return {"sightings": [row_dict(row) for row in rows]}


@app.post("/api/sightings", status_code=201)
async def create_sighting(
    file: UploadFile = File(...),
    species: str = Form(...),
    predicted_species: str = Form(""),
    confidence: float = Form(0.0),
    top_predictions: str = Form("[]"),
    latitude: float = Form(...),
    longitude: float = Form(...),
    observed_at: str = Form(...),
    observer: str = Form(""),
    notes: str = Form(""),
) -> dict:
    species = species.strip()
    if not species or len(species) > 160:
        raise HTTPException(status_code=422, detail="Select a valid species or unknown/review option.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=422, detail="Latitude or longitude is outside its valid range.")
    if not (0 <= confidence <= 1):
        raise HTTPException(status_code=422, detail="Confidence must be between 0 and 1.")
    try:
        parsed_predictions = json.loads(top_predictions)
        if not isinstance(parsed_predictions, list):
            raise ValueError
        stored_predictions = json.dumps(parsed_predictions[:3])
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid top-prediction data.") from exc

    image = await read_valid_image(file)
    filename = f"{uuid.uuid4().hex}.jpg"
    image.save(UPLOAD_DIR / filename, format="JPEG", quality=90, optimize=True)
    created_at = datetime.now(timezone.utc).isoformat()
    source = "field" if predictor.ready else "sample"
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sightings
            (species, predicted_species, confidence, top_predictions, latitude, longitude,
             observed_at, observer, notes, status, source, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                species,
                predicted_species.strip()[:160],
                confidence,
                stored_predictions,
                latitude,
                longitude,
                parse_observed_at(observed_at),
                observer.strip()[:100],
                notes.strip()[:1200],
                source,
                filename,
                created_at,
            ),
        )
        row = connection.execute("SELECT * FROM sightings WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return {"sighting": row_dict(row)}


@app.delete("/api/sightings/samples")
def clear_samples() -> dict:
    with connect() as connection:
        rows = connection.execute("SELECT image_path FROM sightings WHERE source = 'sample'").fetchall()
        deleted = connection.execute("DELETE FROM sightings WHERE source = 'sample'").rowcount
    for row in rows:
        if row["image_path"]:
            (UPLOAD_DIR / row["image_path"]).unlink(missing_ok=True)
    return {"deleted": deleted}


@app.get("/api/sightings/export.csv")
def export_sightings() -> StreamingResponse:
    with connect() as connection:
        rows = [dict(row) for row in connection.execute("SELECT * FROM sightings ORDER BY observed_at DESC")]
    output = io.StringIO()
    fields = [
        "id", "species", "predicted_species", "confidence", "latitude", "longitude",
        "observed_at", "observer", "notes", "status", "source", "created_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bloomwatch_sightings.csv"},
    )


@app.patch("/api/sightings/{sighting_id}")
def update_sighting(sighting_id: int, update: SightingUpdate) -> dict:
    assignments: list[str] = []
    values: list[object] = []
    if update.status is not None:
        assignments.append("status = ?")
        values.append(update.status)
    if update.species is not None:
        species = update.species.strip()
        if not species or len(species) > 160:
            raise HTTPException(status_code=422, detail="Invalid species name.")
        assignments.append("species = ?")
        values.append(species)
    if not assignments:
        raise HTTPException(status_code=422, detail="No changes supplied.")
    values.append(sighting_id)
    with connect() as connection:
        cursor = connection.execute(
            f"UPDATE sightings SET {', '.join(assignments)} WHERE id = ?", values
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Sighting not found.")
        row = connection.execute("SELECT * FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
    return {"sighting": row_dict(row)}


@app.delete("/api/sightings/{sighting_id}")
def delete_sighting(sighting_id: int) -> dict:
    with connect() as connection:
        row = connection.execute("SELECT image_path FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Sighting not found.")
        connection.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
    if row["image_path"]:
        (UPLOAD_DIR / row["image_path"]).unlink(missing_ok=True)
    return {"deleted": sighting_id}


@app.get("/api/stats")
def stats() -> dict:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status = 'verified' THEN 1 ELSE 0 END) AS verified,
                   SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                   COUNT(DISTINCT species) AS species
            FROM sightings
            """
        ).fetchone()
    return {key: (row[key] or 0) for key in ("total", "verified", "pending", "species")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
