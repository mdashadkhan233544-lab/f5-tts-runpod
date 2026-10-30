FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Dhaka

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    transformers==4.38.2 \
    soundfile>=0.12.1 \
    gdown \
    scipy \
    numpy

COPY . .

CMD ["python", "-u", "handler.py"]
