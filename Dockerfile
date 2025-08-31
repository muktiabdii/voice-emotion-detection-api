FROM python:3.12-slim

WORKDIR /app

# Install dependencies untuk audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke container
COPY . .

# Jalankan aplikasi pakai Python, bukan uvicorn langsung
CMD ["python", "main.py"]
