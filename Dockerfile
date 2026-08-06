# Runtime image for the headless CLI. The Textual TUI is interactive and not
# meant for containers. Secrets come from env at run time — never baked in
# (.env is excluded via .dockerignore).
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package. Add optional extras as needed, e.g. `pip install '.[tracing]'`.
# No browser is installed here on purpose: `render_report` falls back to its
# built-in fpdf2 renderer, so PDFs and cover images work in this image.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Drop privileges.
RUN useradd --create-home app && chown -R app /app
USER app

# Default: display CLI help. One-shot examples:
# docker run --rm -e OPENAI_API_KEY=sk-... IMAGE --prompt "hello"
# docker run --rm IMAGE --prompt "hi" --fake # no key
ENTRYPOINT ["financial-research-assistant"]
CMD ["--help"]
