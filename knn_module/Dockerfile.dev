FROM savf/python-27-data-science
RUN mkdir /knn_module
ADD requirements.txt /knn_module
WORKDIR /knn_module
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
