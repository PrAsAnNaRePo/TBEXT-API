# FROM ubuntu:latest
FROM python:3.10-slim-bullseye

# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip libcairo2-dev pkg-config build-essential && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libcairo2-dev pkg-config build-essential ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]