# builder image
FROM python:3.11-slim AS builder
RUN apt-get update && \
    apt-get install -y git

WORKDIR /app

RUN git clone --depth 1 https://github.com/molecule-generator-collection/ChemTSv2.git /app
RUN pip install poetry==1.6.1

COPY ./api_chemtsv2/pyproject.toml ./api_chemtsv2/poetry.lock /app/

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR="/tmp/poetry_cache"

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# runtime image
FROM python:3.11-slim AS runtime
ARG USER_ID
ARG GROUP_ID
ARG CHEMTS_PORT
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV VENV_PATH="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    LISTEN_PORT=${CHEMTS_PORT}
EXPOSE ${CHEMTS_PORT}

COPY --from=builder --chown=${USER_ID}:${GROUP_ID} $VENV_PATH $VENV_PATH

WORKDIR /app

COPY --from=builder --chown=${USER_ID}:${GROUP_ID} /app /app
COPY --chown=${USER_ID}:${GROUP_ID} ./api_chemtsv2/main.py /app

CMD uvicorn main:app --host ${HOST} --port ${LISTEN_PORT}
