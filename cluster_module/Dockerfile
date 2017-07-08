FROM savf/python-27-data-science
RUN mkdir /cluster_module
ADD requirements.txt /cluster_module
WORKDIR /cluster_module
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
