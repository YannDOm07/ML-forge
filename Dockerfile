# ML Forge — image Linux (Python 3.12). Les données vivent dans le volume /data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFORGE_CONFIG=/data/mlforge.toml \
    MLFORGE_ROOT=/data

# Utilisateur non privilégié : le code des expériences ne tourne jamais en root.
RUN useradd --create-home --uid 1000 forge \
    && mkdir -p /data /home/forge/.kaggle \
    && chown -R forge:forge /data /home/forge/.kaggle

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY --chown=forge:forge pyproject.toml README.md AGENTS.md docker-entrypoint.sh ./
COPY --chown=forge:forge forge ./forge
COPY --chown=forge:forge tests ./tests
COPY --chown=forge:forge docs ./docs
RUN chmod +x /app/docker-entrypoint.sh

USER forge
VOLUME ["/data"]
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]
