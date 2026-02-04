# AsgardBench Dockerfile
# Runs the benchmark evaluation in a containerized environment

FROM python:3.11-slim

# Install system dependencies for AI2-THOR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY AsgardBench/ ./AsgardBench/
COPY Generated/ ./Generated/
COPY .env.example ./

# Install dependencies
RUN uv sync --frozen

# Create output directory
RUN mkdir -p /app/Test

# Default environment variables (override with -e)
ENV OPENAI_API_KEY=""
ENV OPENAI_BASE_URL="https://api.openai.com/v1"
ENV ASGARDBENCH_TEST_DIR="/app/Test"

# Default command: show help
ENTRYPOINT ["uv", "run", "python", "-m", "AsgardBench.Model.model_tester"]
CMD ["--help"]

# Example usage:
# docker build -t asgardbench .
# docker run -e OPENAI_API_KEY=sk-... asgardbench --test magt_benchmark_sanity --model gpt-4o
# docker run -v $(pwd)/results:/app/Test -e OPENAI_API_KEY=sk-... asgardbench --test magt_benchmark --model gpt-4o
