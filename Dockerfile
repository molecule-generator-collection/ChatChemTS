# builder image
FROM python:3.11-slim AS builder

RUN pip install poetry==1.6.1

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR="/tmp/poetry_cache"

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# runtime image
FROM python:3.11-slim AS runtime
RUN apt-get update && \
    apt-get install -y libxrender1 libxext6

ENV VENV_PATH="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    LISTEN_PORT=8501
EXPOSE 8501

COPY --from=builder $VENV_PATH $VENV_PATH

WORKDIR /app

COPY ./chatmolgen.py /app/
COPY ./pages/ /app/pages/

CMD ["streamlit", "run", "chatmolgen.py"]

