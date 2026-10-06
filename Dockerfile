# syntax=docker/dockerfile:1
# Dockerfile for an image which includes a PISA installation through the fast uv Python package manager.
# This image is based on a "slimmified" Debian 13 image.

# Docker: https://docs.docker.com/get-started/docker-concepts/building-images/
# uv: https://github.com/astral-sh/uv

FROM debian:trixie-slim

# Install curl (for uv installer) & git + gcc (for building PISA) & uv from standalone installer
# (uses "heredocs": https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/)
RUN <<EOT
apt-get update
apt-get install -y gcc curl git
UV_NO_MODIFY_PATH=1 curl -Lf https://astral.sh/uv/install.sh | sh
EOT

# Add uv executable to path and set PISA path (will remain available in container)
ENV PATH=/root/.local/bin:$PATH PISA=pisa/

# Create PISA source folder
RUN mkdir -p $PISA

# Link pisa folder
COPY . $PISA

# Create virtual env. with Python 3.14, activate it, and install PISA (non-editable is important with uv here) + jupyter notebook
RUN <<EOT
uv venv --python 3.14 .venv
. .venv/bin/activate
uv pip install $PISA
uv pip install notebook
EOT

# Expose the Jupyter server port
EXPOSE 8888

# Also pre-activate the virtual environment for the container (prepend binary directory to PATH) so user needn't do so
ENV PATH=/.venv/bin:$PATH
