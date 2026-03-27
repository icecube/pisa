# Dockerfile for an image which includes a PISA installation through the fast uv Python package manager.
# This image is based on a "slimmified" Debian 13 image.

# Docker: https://docs.docker.com/get-started/docker-concepts/building-images/
# uv: https://github.com/astral-sh/uv

FROM debian:trixie-slim

# Install curl (for uv installer) & git + gcc (for building PISA) & uv from standalone installer
RUN apt-get update && apt-get install -y gcc curl git && curl -Lf https://astral.sh/uv/install.sh | sh

# Add executable to path and set PISA path
ENV PATH=/root/.local/bin/:$PATH PISA=pisa/

# Create PISA source folder
RUN mkdir -p $PISA

# Link pisa folder
COPY . $PISA

# Create virtual env. with Python 3.12, activate it, and install PISA (non-editable is important with uv here) + jupyter notebook
RUN uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install $PISA && uv pip install notebook

# Expose the Jupyter server port
EXPOSE 8888
