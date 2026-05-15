import datetime
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

MODELS_DIR = Path(__file__).parent / "models"
ARTIFACT_FILENAMES = {
    "preprocessing": "preprocessing.pkl",
    "filtering": "filtering.pkl",
    "model": "model.pkl",
}

def _validate_artifacts(preprocessing: Any, filtering: Any, model: Any) -> None:
    if not hasattr(preprocessing, "transform"):
        raise HTTPException(status_code=400, detail="preprocessing debe tener método transform")
    

    if not hasattr(filtering, "transform"):
        raise HTTPException(status_code=400, detail="filtering debe tener método transform")
    
    if isinstance(model, dict):
        print("MODEL KEYS:", model.keys())
        model = model["model"]

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=400, detail="model debe tener método predict_proba")

    
def _load_artifact(filename: str) -> Any:
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[warn] no se pudo cargar {path.name}: {e}")
        return None


preprocessor = _load_artifact(ARTIFACT_FILENAMES["preprocessing"])
filter_step = _load_artifact(ARTIFACT_FILENAMES["filtering"])
model = _load_artifact(ARTIFACT_FILENAMES["model"])

try:
    _validate_artifacts(preprocessor, filter_step, model)
except Exception as e:
    print(f"[warn] artefactos no cargados o inválidos: {e}")
    preprocessor = None
    filter_step = None
    model = None


@app.post("/predict/")
async def predict(data: dict[str, Any] | list[dict[str, Any]]):
    if preprocessor is None or filter_step is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelos no cargados. Sube los .pkl con /model/upload.",
        )

    try:
        is_batch = isinstance(data, list)
        df = pd.DataFrame(data if is_batch else [data])

        X_pre, _ = preprocessor.transform(df)
        X_filt = filter_step.transform(X_pre)

        proba = model.predict_proba(X_filt)
        p_default_arr = proba[:, 1]

        _, p0p1 = model.predict_proba(X_filt, p0_p1_output=True)
        p0p1 = np.asarray(p0p1)

        p_low_arr = p0p1[:, :, 0].mean(axis=0)
        p_high_arr = p0p1[:, :, 1].mean(axis=0)

        rows = []

        for i in range(len(df)):
            p_default = float(p_default_arr[i])
            p_low = float(p_low_arr[i])
            p_high = float(p_high_arr[i])

            interval_width = p_high - p_low
            decision = "agent" if interval_width > 0.2 else "auto"
            reason = "p_high - p_low > 0.2" if decision == "agent" else "p_high - p_low <= 0.2"

            rows.append({
                "p_default": p_default,
                "p_low": p_low,
                "p_high": p_high,
                "decision": decision,
                "reason": reason,
            })

        return rows if is_batch else rows[0]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/model/upload")
async def upload_model(
    preprocessing: UploadFile = File(...),
    filtering: UploadFile = File(...),
    model_file: UploadFile = File(..., alias="model"),
):
    global model
    global preprocessor
    global filter_step

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    uploads = {
        "preprocessing": preprocessing,
        "filtering": filtering,
        "model": model_file,
    }

    saved: dict[str, dict[str, Any]] = {}
    loaded: dict[str, Any] = {}

    for key, upload in uploads.items():
        if not upload.filename or not upload.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail=f"'{key}' debe ser un fichero .pkl (recibido: {upload.filename})",
            )

        destination = MODELS_DIR / ARTIFACT_FILENAMES[key]
        contents = await upload.read()

        try:
            loaded[key] = pickle.loads(contents)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo cargar '{key}': {e}",
            )

        destination.write_bytes(contents)
        saved[key] = {"path": str(destination), "size": len(contents)}

    loaded_model = loaded["model"]

    if isinstance(loaded_model, dict) and "model" in loaded_model:
        loaded_model = loaded_model["model"]

    _validate_artifacts(
        loaded["preprocessing"],
        loaded["filtering"],
        loaded_model,
    )

    preprocessor = loaded["preprocessing"]
    filter_step = loaded["filtering"]
    model = loaded_model

    timestamp = datetime.datetime.now().isoformat()

    return {
        "status": "ok",
        "saved": saved,
        "timestamp": timestamp,
    }