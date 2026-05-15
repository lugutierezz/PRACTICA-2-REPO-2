import datetime
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

print("USANDO ESTE MAIN:", __file__)

MODELS_DIR = Path(__file__).parent / "models"
ARTIFACT_FILENAMES = {
    "preprocessing": "preprocessing.pkl",
    "filtering": "filtering.pkl",
    "model": "model.pkl",
}


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
filter = _load_artifact(ARTIFACT_FILENAMES["filtering"])
model = _load_artifact(ARTIFACT_FILENAMES["model"])


@app.post("/predict/")
async def predict(data: dict[str, Any] | list[dict[str, Any]]):
    # data es el JSON que llega en el body: una fila (dict) o un batch (list[dict])
    if preprocessor is None or filter is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelos no cargados. Sube los .pkl con /upload_model/.",
        )

    is_batch = isinstance(data, list)
    df = pd.DataFrame(data if is_batch else [data])

    print("1 df creado", df.shape)

    X_pre, _ = preprocessor.transform(df)
    print("2 preprocessor ok", type(X_pre), getattr(X_pre, "shape", None))

    X_filt = filter.transform(X_pre)
    print("3 filter ok", type(X_filt), getattr(X_filt, "shape", None))

    proba = model.predict_proba(X_filt)
    p_default_arr = proba[:, 1]
    print("4 predict_proba ok", type(proba), getattr(proba, "shape", None))

    _, p0p1 = model.predict_proba(X_filt, p0_p1_output=True)
    print("5 predict_proba p0p1 ok", type(p0p1))
    p0p1 = np.asarray(p0p1)

    p_low_arr = p0p1[:, :, 0].mean(axis=0)
    p_high_arr = p0p1[:, :, 1].mean(axis=0)

    rows =   []       
    for i in range(len(df)):
            p_default = float(p_default_arr[i])
            p_low = float(p_low_arr[i])
            p_high = float(p_high_arr[i])

            interval_width = p_high - p_low
            decision = "agent" if interval_width > 0.2 else "auto"
            reason = "p_high - p_low > 0.2" if decision == "agent" else "p_high - p_low <= 0.2"

            rows.append(
                {
                    "p_default": p_default,
                    "p_low": p_low,
                    "p_high": p_high,
                    "decision": decision,
                    "reason": reason,
                }
            )
    return rows if is_batch else rows[0]


@app.post("/upload_model/")
async def upload_model(
    preprocessing: UploadFile = File(...),
    filtering: UploadFile = File(...),
    model_file: UploadFile = File(..., alias="model"),
):
    global model
    global preprocessor
    global filter

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    uploads = {
        "preprocessing": preprocessing,
        "filtering": filtering,
        "model": model_file,
    }
    saved: dict[str, dict[str, Any]] = {}
    loaded: dict[str, Any] = {}
    for key, upload in uploads.items():
        if not upload.filename.endswith(".pkl"):
            raise HTTPException(
                status_code=400,
                detail=f"'{key}' debe ser un fichero .pkl (recibido: {upload.filename})",
            )
        
        destination = MODELS_DIR / ARTIFACT_FILENAMES[key]
        contents = await upload.read()
        destination.write_bytes(contents)
        loaded[key] = pickle.loads(contents)
        saved[key] = {"path": str(destination), "size": len(contents)}

    preprocessor = loaded["preprocessing"]
    filter = loaded["filtering"]
    model = loaded["model"]

    return {"status": "ok", "saved": saved,  "version": datetime.datetime.now().isoformat(),}