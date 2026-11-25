# Image for the parity check: the sandbox's Python and coverage tools, plus tqdm,
# which the reference eval_overall.py imports.
FROM python:3.12.16-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

COPY src/inspect_testeval/sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && pip install --no-cache-dir tqdm==4.70.1 \
    && rm /tmp/requirements.txt

RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent
