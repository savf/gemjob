FROM python:2.7-alpine
RUN mkdir /cluster_module
ADD requirements.txt /cluster_module
WORKDIR /cluster_module
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
