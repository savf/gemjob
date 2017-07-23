FROM savf/python-27-data-science
RUN mkdir /feedback_module
ADD requirements.txt /feedback_module
WORKDIR /feedback_module
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
