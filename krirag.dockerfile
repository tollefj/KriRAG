# FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

COPY src/requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ /app
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 install.py

EXPOSE 8501
CMD ["bash", "-c", "streamlit run ui.py"]