# Sandbox for running model-generated tests against the program under test.
# Base image pinned by digest; Python packages installed from a hashed lock
# (requirements.txt, generated from requirements.in).
FROM python:3.12.16-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent

CMD ["tail", "-f", "/dev/null"]
