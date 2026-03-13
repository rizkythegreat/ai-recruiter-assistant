# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face Spaces uses port 7860
# Update the command to use the $PORT environment variable if needed, 
# but HF defaults to 7860.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]