# ## Executive summary (read this first)
# Frozen v1 forecasting code only. No units, predictions, keys or model weights.
# Build-time dependencies; read-only runtime with organizer-provided House API.
FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26
LABEL qfbench2.interface_version="2.0" qfbench2.track="forecasting" qfbench2.verb="forecast"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp XDG_CACHE_HOME=/tmp/cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.5.3 pandas==2.3.3 pyarrow==25.0.1 scipy==1.18.1 statsforecast==2.1.1
COPY requirements.txt /opt/agent/requirements.txt
RUN pip install --no-cache-dir -r /opt/agent/requirements.txt && pip check
COPY pipelines/ /opt/agent/pipelines/
COPY scripts/ /opt/agent/scripts/
COPY forecast.py freeze.json LICENSE /opt/agent/
RUN printf '#!/bin/sh\nexec python3 /opt/agent/forecast.py "$@"\n' > /usr/local/bin/forecast && chmod 755 /usr/local/bin/forecast
USER 1000:1000
WORKDIR /tmp
CMD ["forecast", "--help"]
