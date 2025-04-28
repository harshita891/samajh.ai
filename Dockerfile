FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY detector/ ./detector/
COPY tracker/ ./tracker/
COPY utils/ ./utils/
COPY app/ ./app/

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "app/main.py"]
