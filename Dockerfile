FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir gym numpy

CMD ["python", "app.py"]FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir gym numpy

CMD ["python", "app.py"]