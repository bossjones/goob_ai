# ARG VARIANT=3.10.14
# FROM python:${VARIANT} AS builder

# ENV PYTHONDONTWRITEBYTECODE=True

# WORKDIR /opt
# COPY pyproject.toml requirements.lock requirements-dev.lock ./

# # hadolint ignore=DL3013,DL3042
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.lock && \
#     pip install --no-cache-dir -r requirements-dev.lock && \


# FROM python:${VARIANT}-slim
# COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages

# ENV PYTHONUNBUFFERED=True

# WORKDIR /
