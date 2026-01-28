# Vi bruker samme base som sist, men sikrer at vi har nødvendige system-avhengigheter
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Unngå spørsmål under installasjon
ENV DEBIAN_FRONTEND=noninteractive

# Installer system-biblioteker for lyd (viktig for soundfile/librosa)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Kopier requirements og installer python-pakker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier kildekoden din
COPY src/ /workspace/src/

# Gjør script kjørbare
RUN chmod +x /workspace/src/entrypoint.sh

# Opprett mapper for data
RUN mkdir -p /workspace/data /workspace/output

# Entrypoint styrer showet
ENTRYPOINT ["/workspace/src/entrypoint.sh"]
