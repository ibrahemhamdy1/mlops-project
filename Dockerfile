# syntax=docker/dockerfile:1
# ---------- build stage: dependencies only ----------
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim AS runtime
WORKDIR /app

# non-root user (DevSecOps: never run as root)
RUN groupadd -r app && useradd -r -g app app

COPY --from=deps /install /usr/local
COPY src/ ./src/
COPY model/ ./model/

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
