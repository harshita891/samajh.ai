FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --upgrade onnxruntime

COPY main.py .
COPY models/ /app/models/
COPY tracker/ /app/tracker/
COPY detector/ /app/detector/
COPY utils/ /app/utils/
COPY configs/ /app/configs/

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py", "0"]