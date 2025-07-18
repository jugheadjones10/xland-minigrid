FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Build argument for installing CI dependencies
# ARG INSTALL_CI=false

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Make "python" reference python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN pip install --upgrade pip && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# ----------------- ADD THIS STEP -----------------
# Explicitly install CUDA-enabled JAX using pip. This is the most reliable method.
# It uses the special "cuda12_pip" extra and finds the correct wheel from Google's repo.
RUN pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# -------------------------------------------------


# Copy only the files needed to install dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY LICENSE LICENSE
COPY README.md README.md
# src needs to come before poetry install since we are installing the current project as well
COPY ./src /src

# Configure poetry to not create a virtual env
RUN poetry config virtualenvs.create false

# Install dependencies
# RUN if [ "$INSTALL_CI" = "true" ]; then \
#         poetry install --extras "baselines ci"; \
#     else \
#         poetry install --extras "baselines"; \
#     fi

RUN poetry install --extras baselines

COPY ./training /training