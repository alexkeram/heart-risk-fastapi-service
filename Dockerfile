# 1. Base image Python 3.12
FROM python:3.12-slim

# 2. Acceleration for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Requirements for scikit
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Work directory
WORKDIR /app

# 5. Requirements for the cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy app code
COPY app app
COPY src src
COPY artifacts artifacts
COPY README.md README.md

# 7. Port for the app
EXPOSE 8000

# 8. Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
