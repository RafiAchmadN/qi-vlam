# ---- build stage ----
FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---- runtime stage ----
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /install /usr/local
COPY src/ src/
COPY scripts/ scripts/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "uvicorn", "--factory", "src.inference.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
