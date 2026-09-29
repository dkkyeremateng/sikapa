# Runtime image for the headless CLI and the always-on service (`--serve`). The
# Textual TUI is interactive and not meant for containers — reach it over SSH
# instead (see deploy/README.md). Secrets come from env at run time — never baked
# in (.env is excluded via .dockerignore).
FROM python:3.13-slim

# Optional extras to install, comma-separated. `anthropic` is the default because
# it is the provider this deployment signs in with; add `tracing`, `ofx`,
# `documents`, `google`, `groq` as needed:
#   docker build --build-arg EXTRAS=anthropic,tracing -t fra .
ARG EXTRAS=anthropic

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# Install from the LOCK, not a fresh resolve. `pip install .` alone resolves the
# ranges in pyproject.toml against today's index, which is how mcp 2.x once
# slipped in and broke the broker adapter on import while locked installs were
# fine; exporting uv.lock gives the image the exact versions the suite ran on.
COPY pyproject.toml uv.lock ./
RUN set -eu; \
    pip install uv; \
    extras=""; \
    for e in $(echo "$EXTRAS" | tr ',' ' '); do extras="$extras --extra $e"; done; \
    uv export --frozen --no-dev --no-emit-project --no-hashes $extras -o /tmp/requirements.txt; \
    pip install -r /tmp/requirements.txt; \
    pip uninstall -y uv; \
    rm /tmp/requirements.txt

COPY src ./src
RUN pip install --no-deps .

# Drop privileges. A fixed uid so the host's data directory can be chowned to
# match it before the first start (deploy/README.md, "First start").
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.financial-research-assistant \
    && chown -R app /home/app /app
USER app

# Everything the agent knows lives here: statements, journal, memory, tasks,
# reports, the investor profile. Mount a host directory over it — without one,
# a container restart forgets the lot.
VOLUME /home/app/.financial-research-assistant

# Healthy = the service's job loop ticked recently. `--status --check` reads the
# heartbeat file and exits non-zero when it has gone stale.
HEALTHCHECK --interval=2m --timeout=30s --start-period=3m --retries=3 \
    CMD ["financial-research-assistant", "--status", "--check"]

# Default: display CLI help. Examples:
# docker run --rm -e OPENAI_API_KEY=sk-... IMAGE --prompt "hello"
# docker run --rm IMAGE --prompt "hi" --fake # no key
# docker run -d --env-file /etc/fra/env -v /srv/fra/data:/home/app/.financial-research-assistant IMAGE --serve
ENTRYPOINT ["financial-research-assistant"]
CMD ["--help"]
