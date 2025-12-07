FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# git is often required for installing dependencies from git repositories
# libgl1 and libglib2.0-0 are sometimes needed for opencv or other image libraries if used by gradio dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
# Including protobuf explicitly as it was needed earlier
RUN pip install --no-cache-dir -r requirements.txt protobuf

# Copy the rest of the application
COPY . .

# Set environment variables
# HF_HOME defines where Hugging Face downloads models. We will map a volume here.
ENV HF_HOME="/data/huggingface"
# Create the directory
RUN mkdir -p $HF_HOME

# Expose the port Gradio runs on
EXPOSE 7860

# The command to run the application
# We use python -u for unbuffered output so logs show up immediately
CMD ["python", "-u", "app.py"]
