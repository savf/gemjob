FROM ubuntu:16.04

RUN apt-get update -qqy \
    && apt-get -qqy --no-install-recommends install \
    ca-certificates \
    apt-utils \
    wget \
    unzip \
    xvfb \
    ratpoison \
    x11vnc \
    python-pip \
    python-dev \
    build-essential \
    locales
RUN apt-get --reinstall install -qqy xfonts-base

#===============
# Google Chrome
#===============
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
  && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update -qqy \
  && apt-get -qqy install \
    google-chrome-stable \
  && rm /etc/apt/sources.list.d/google-chrome.list \
  && rm -rf /var/lib/apt/lists/*

#==================
# Chrome webdriver
#==================
ENV CHROME_DRIVER_VERSION 2.29
RUN wget --no-verbose -O /tmp/chromedriver_linux64.zip http://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip \
  && rm -rf /opt/selenium/chromedriver \
  && unzip /tmp/chromedriver_linux64.zip -d /opt/selenium \
  && rm /tmp/chromedriver_linux64.zip \
  && mv /opt/selenium/chromedriver /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION \
  && chmod 755 /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION \
  && ln -fs /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION /usr/bin/chromedriver

#=================================
# Chrome Launch Script Modification
#=================================
COPY chrome_launcher.sh /opt/google/chrome/google-chrome

#=================
# Locale settings
#=================
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
RUN locale-gen en_US.UTF-8 \
  && dpkg-reconfigure --frontend noninteractive locales \
  && apt-get update -qqy \
  && apt-get -qqy --no-install-recommends install \
    language-pack-en \
  && rm -rf /var/lib/apt/lists/*

#=======
# Fonts
#=======
RUN apt-get update -qqy \
  && apt-get -qqy --no-install-recommends install \
    fonts-ipafont-gothic \
    xfonts-100dpi \
    xfonts-75dpi \
    xfonts-cyrillic \
    xfonts-scalable \
  && rm -rf /var/lib/apt/lists/*

#==========
# Selenium
#==========
RUN pip install --upgrade pip
RUN pip install selenium setuptools

#============================
# Clean up
#============================
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.config/google-chrome
RUN chmod +x /opt/google/chrome/google-chrome

RUN mkdir /crawler_module
WORKDIR /crawler_module
ADD requirements.txt /crawler_module
RUN pip install -r requirements.txt

ENTRYPOINT ["/crawler_module/entrypoint.sh"]
