FROM python:3.8.1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "main.py"]