# Image for the one-command reproduction (docker-compose.yml): the pipeline on
# the first 200 frames of KITTI 00, then scripts/docker_report.py.
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install .

ENTRYPOINT ["bash", "scripts/docker_entrypoint.sh"]
