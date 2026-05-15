# model_serving

Proyecto sencillo de **FastAPI** para servir un modelo de Machine Learning.

## Estructura

```
Practica2Repo2
model_serving/
├── main.py            
├── models/
    ├── filtering.pkl
    ├── preorocesing.pkl
    ├── model.pkl
├── src (contiene base_filtering y base_preprocessing de la prcatica anterior)   
└── README.md
```

## Instalación

Este proyecto usa [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Ejecución

```bash
uv run uvicorn main:app --reload
```

API en `http://127.0.0.1:8000` y documentación interactiva en `http://127.0.0.1:8000/docs`.

## Endpoint

