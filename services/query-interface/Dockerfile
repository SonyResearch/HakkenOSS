FROM python:3.11-slim AS builder

# Installing UV (TODO: Build another image with UV)
# The installer requires curl (and certificates) to download the release archive
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    apt-get update -y && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"


WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN --mount=type=secret,id=codeartifact_username,uid=1000 \
    --mount=type=secret,id=codeartifact_password,uid=1000 \
    export UV_INDEX_HAKKEN_CODEARTIFACT_REGISTRY_USERNAME=$(cat /run/secrets/codeartifact_username) ; \
    export UV_INDEX_HAKKEN_CODEARTIFACT_REGISTRY_PASSWORD=$(cat /run/secrets/codeartifact_password) ; \
    uv sync --no-dev

FROM python:3.11-slim

ENV APP_DIR=/app

COPY --from=builder /app/.venv ${APP_DIR}/.venv

ENV PATH="${APP_DIR}/.venv/bin:$PATH"

COPY ./service.py ./service.py
CMD [ "-m", "service" ]
