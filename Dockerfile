FROM python:3.11-slim

# curl is only used by the HEALTHCHECK below. cryptography/bcrypt ship
# prebuilt wheels for this platform, so no compiler toolchain is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Never bake secrets or local state into the image — .dockerignore excludes
# .env, users.yaml, storage/, and app_cache/ from the build context entirely.
# Pass secrets at container-start time instead, e.g.:
#   docker run --env-file .env -p 8501:8501 pathex
RUN useradd --create-home appuser \
    && mkdir -p storage app_cache \
    && chown -R appuser:appuser /app
USER appuser

# storage/ (user accounts) and app_cache/ (extraction cache) should be mounted
# from the host if you want them to survive a container restart/redeploy —
# e.g. `docker run -v pathex_storage:/app/storage -v pathex_cache:/app/app_cache ...`
VOLUME ["/app/storage", "/app/app_cache"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]
