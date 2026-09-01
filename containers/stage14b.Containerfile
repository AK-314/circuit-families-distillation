# syntax=docker/dockerfile:1.7
# Both bases are mandatory immutable name@sha256:digest values supplied by the builder.
ARG PYTHON_BASE_IMAGE
ARG UV_BASE_IMAGE
FROM ${UV_BASE_IMAGE} AS uv-source
FROM ${PYTHON_BASE_IMAGE} AS runtime
COPY --from=uv-source /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
WORKDIR /workspace
COPY .python-version pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev
ENV PATH="/workspace/.venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "circuit_families.stage14b.cli"]
