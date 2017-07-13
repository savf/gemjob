FROM savf/python-27-data-science
RUN mkdir /jobtype_module
ADD requirements.txt /jobtype_module
WORKDIR /jobtype_module
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
