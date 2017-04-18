FROM python:2.7-alpine
ADD requirements.txt /data_module
WORKDIR /data_module
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
