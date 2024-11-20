FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="tollefj"
ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt update && \
    apt install -y git python3 python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
ARG PYTORCH='2.5.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
ARG CUDA='cu121'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; \
    python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

COPY src/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt && \
    rm /app/requirements.txt

COPY src/ /app
RUN python3 /app/install.py && \ rm /app/install.py
RUN rm -rf /root/.cache

EXPOSE 8501
CMD ["bash", "-c", "streamlit run /app/ui.py"]