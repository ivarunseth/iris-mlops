FROM python:3.11-slim AS base

ARG UID=1000
ARG GID=1000
ARG USER=appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && apt-get clean \
    && groupadd -g ${GID} ${USER} \
    && useradd -u ${UID} -g ${GID} -m ${USER} \
    && mkdir -p /home/${USER}/.cache/pip /app \
    && chown -R ${USER}:${USER} /home/${USER} /app \
    && chmod 755 /app \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man

USER ${USER}

WORKDIR /app

ENV PATH="/home/${USER}/.local/bin:${PATH}:" \
    PYTHONPATH="."

COPY --chown=${USER}:${USER} requirements.txt .

RUN python -m pip install -U pip setuptools wheel \
    && pip install --user -r requirements.txt

ENV FLASK_ENV="production" \
    FLASK_APP="app.py" 

COPY --chown=${USER}:${USER} . .

EXPOSE 5001

CMD ["gunicorn", "-b", "0.0.0.0:5001", "-w", "1", "--access-logfile", "-", "--log-level", "info", "app:application"]
