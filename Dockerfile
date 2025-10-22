FROM python:3.11.0

# Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=false \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# System deps (optional): uncomment if you want to enable microphone recording
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libportaudio2 \
#  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE ${PORT}

CMD ["python", "app.py"]

