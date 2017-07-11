FROM savf/python-27-data-science
RUN mkdir /budget_module
ADD requirements.txt /budget_module
WORKDIR /budget_module
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
