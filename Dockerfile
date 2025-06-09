# -------- Dockerfile --------
# 1. Choose a tiny, up-to-date base image with Python 3.12
FROM python:3.12-slim

# 2. Install build tools that some wheels still need at install-time
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Set a non-root user (optional but good practice)
ARG USER=appuser
RUN useradd -m ${USER}
USER ${USER}

# 4. Create and set working directory in the container
WORKDIR /home/${USER}/app

# 5. Copy dependency list first (better layer-cache utilisation)
COPY --chown=${USER}:${USER} requirements.txt .

# 6. Install Python deps WITH pip cache kept in layer
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the repo (scripts/, src/, etc.)
COPY --chown=${USER}:${USER} . .

# 8. Default command – opens a shell; override in `docker run`
CMD ["/bin/bash"]
# ----------------------------
