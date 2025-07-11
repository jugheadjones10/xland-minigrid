FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Make "python" reference python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy only the files needed to install dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY LICENSE LICENSE
COPY README.md README.md
COPY ./src /src

# Install Poetry
RUN pip install --upgrade pip && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Install dependencies
RUN poetry install --extras baselines


RUN mkdir training && touch training/__init__.py


# I want to eventually install the current project as well, which requires the README.md
RUN poetry install --extras baselines

COPY ./training /training

# poetry run python rl_for_dummies/pushworld/meta_a2c_pushworld.py