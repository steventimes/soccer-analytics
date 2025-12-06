FROM ubuntu:24.04

# Set working directory inside container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    build-essential \
    pkg-config \
    libpq-dev \
    libdbus-1-dev \
    libdbus-glib-1-dev \
    libgirepository1.0-dev \
    libsystemd-dev \
    libcairo2-dev \
    curl \
    wget \
    git \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirement.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8000

CMD ["tail", "-f", "/dev/null"]