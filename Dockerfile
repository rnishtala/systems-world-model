FROM python:3.11-slim AS control
WORKDIR /app
COPY launcher/requirements.txt /app/control-requirements.txt
RUN pip install --no-cache-dir -r /app/control-requirements.txt
COPY launcher /app/launcher
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
CMD ["python", "-m", "launcher.server"]

FROM control AS collector
COPY microservice/__init__.py microservice/live.py microservice/temporal_pilot.py microservice/queue_drain.py /app/microservice/

FROM python:3.11-slim AS runner
WORKDIR /app
COPY requirements-cpu.txt /app/requirements-cpu.txt
RUN pip install --no-cache-dir -r /app/requirements-cpu.txt
ADD https://github.com/danijar/dreamerv3/archive/e01491fad6434b2245a3b8ca201dd7faedcc458c.tar.gz /tmp/upstream.tar.gz
RUN mkdir /app/upstream && tar xzf /tmp/upstream.tar.gz -C /app/upstream --strip-components=1 && rm /tmp/upstream.tar.gz
COPY microservice /app/microservice
COPY launcher /app/launcher
ENV DREAMER_SOURCE=/app/upstream PYTHONPATH=/app:/app/upstream PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XDG_CACHE_HOME=/tmp/cache
ENV XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
USER 1000:1000
ENTRYPOINT []
