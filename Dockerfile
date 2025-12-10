FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed to compile some Python wheels if necessary
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget libatlas3-base git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy source
COPY . /app

# Use PORT env var (Render provides $PORT). Default to 8501 for local runs.
ENV PORT=8501
EXPOSE ${PORT}

# Start Streamlit listening on the provided port
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
