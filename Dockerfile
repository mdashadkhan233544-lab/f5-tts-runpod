FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies ensuring compatible torchaudio
RUN pip install --no-cache-dir \
    runpod \
    torchaudio==2.4.0 \
    f5-tts \
    soundfile

COPY handler.py /app/handler.py

CMD [ "python", "-u", "/app/handler.py" ]
