FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    f5-tts \
    soundfile \
    torchaudio

RUN python -c "from f5_tts.api import F5TTS; F5TTS(model_type='F5-TTS', device='cpu')"

COPY handler.py /app/handler.py

CMD [ "python", "-u", "/app/handler.py" ]
