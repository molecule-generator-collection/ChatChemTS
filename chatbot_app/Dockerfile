# builder image
FROM python:3.11-slim AS builder

RUN pip install poetry==1.6.1

WORKDIR /app

COPY ./chatbot_app/pyproject.toml ./chatbot_app/poetry.lock /app/

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR="/tmp/poetry_cache"

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# runtime image
FROM python:3.11-slim AS runtime
ARG USER_ID
ARG GROUP_ID
ARG CHATBOT_PORT

ENV VENV_PATH="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    LISTEN_PORT=${CHATBOT_PORT} \
    CHAINLIT_PORT=${CHATBOT_PORT}
EXPOSE ${CHATBOT_PORT}

COPY --from=builder --chown=${USER_ID}:${GROUP_ID} $VENV_PATH $VENV_PATH

WORKDIR /app
RUN chown -R ${USER_ID}:${GROUP_ID} /app

COPY --chown=${USER_ID}:${GROUP_ID} ./chatbot_app/app.py ./chatbot_app/chainlit.md ./chatbot_app/prompts.py ./chatbot_app/tools.py ./chatbot_app/util.py /app/
COPY --chown=${USER_ID}:${GROUP_ID} ./chatbot_app/.chainlit/ /app/.chainlit/
COPY --chown=${USER_ID}:${GROUP_ID} ./chatbot_app/public/ /app/public

CMD ["chainlit", "run", "app.py"]
