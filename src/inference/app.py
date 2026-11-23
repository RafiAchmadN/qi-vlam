"""Uvicorn factory entrypoint."""
from src.inference.model_server import build_app


def create_app():
    return build_app()
