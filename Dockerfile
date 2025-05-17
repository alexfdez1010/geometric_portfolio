FROM python:3.12-slim

WORKDIR /app

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies using pip directly from pyproject.toml
# This is more reliable than trying to get uv working in the container
RUN pip install --no-cache-dir -e .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "src/geometric_portfolio/app.py", "--server.address=0.0.0.0"]
