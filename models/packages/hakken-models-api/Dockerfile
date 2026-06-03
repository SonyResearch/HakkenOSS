# TODO: This dockerfile is basically copied from kge_api
FROM python:3.11-slim AS builder

# Install UV (TODO: Build another image with UV)
# The installer requires curl (and certificates) to download the release archive
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/0.8.14/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"                          

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY README.md ./README.md


RUN --mount=type=secret,id=codeartifact_username,uid=1000 \
    --mount=type=secret,id=codeartifact_password,uid=1000 \
    sh -c '\
      UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME="$(cat /run/secrets/codeartifact_username)" \
      UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD="$(cat /run/secrets/codeartifact_password)" \
      uv sync --no-dev --no-install-project'


FROM python:3.11-slim

ENV APP_DIR=${HAKKEN_HOME}/app

COPY --from=builder --chown=${HAKKEN_USER}:HAKKEN_GROUP /app/.venv ${APP_DIR}/.venv

ENV PATH="${APP_DIR}/.venv/bin:$PATH"

# Switch to the new user
USER ${HAKKEN_USER}

COPY ./config ./config
COPY ./hakken-models-api ./hakken-models-api
COPY ./service.py ./service.py
COPY init.sh ./init.sh

RUN ./init.sh

CMD ["-m", "service"]
