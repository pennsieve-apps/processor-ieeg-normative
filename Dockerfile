FROM --platform=linux/amd64 python:3.12.15-slim-bookworm

WORKDIR /processor

# Pinned Python dependencies (binary wheels only, no apt packages), plus the Lambda runtime
# interface client so the same image runs as a Lambda function.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d \( -name __pycache__ -o -name tests \) -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# One BLAS thread: the cross-spectra are many small matrix products, and OpenBLAS sizes its pool
# from the host's CPUs, not the container's quota. With 2 vCPUs (emulated) the atlas coherence took
# 50 s with the default pool and 1.5 s with one thread.
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

# Runs main() once as a container, or serves processor.handler as a Lambda function
# (see processor/entry.py).
ENTRYPOINT ["python3", "-m", "processor.entry"]
