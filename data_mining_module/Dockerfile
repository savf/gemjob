FROM serebrov/python27-pandas
RUN mkdir /data_mining_module
ADD requirements.txt /data_mining_module
WORKDIR /data_mining_module
RUN apt-get update && \
    apt-get install -y libfreetype6-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/include/freetype2/ft2build.h /usr/include/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
