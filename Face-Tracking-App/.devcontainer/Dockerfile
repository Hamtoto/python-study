FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY build_20250710-1_amd64.deb /tmp/opencv.deb
RUN dpkg -i /tmp/opencv.deb || \
    apt-get install -f -y && \
    rm /tmp/opencv.deb

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .