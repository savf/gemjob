FROM python:2.7-alpine
RUN mkdir /webserver_module
ADD requirements.txt /webserver_module
WORKDIR /webserver_module
RUN pip install -r requirements.txt
CMD ["sh", "django_entrypoint.sh"]
