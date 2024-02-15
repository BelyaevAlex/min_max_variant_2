FROM python:3.10
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /var/www/5scontrol
COPY . .
RUN mkdir -p /usr/src/app
ENTRYPOINT ["python", "-u", "MinMaxAlgoritm.py"]