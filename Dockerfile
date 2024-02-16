FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /var/www/5scontrol
COPY . .

ENTRYPOINT ["python", "-u", "MinMaxAlgoritm.py"]