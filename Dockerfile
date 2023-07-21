FROM ubuntu

WORKDIR /resspect
ENV HOME /
RUN echo "entering resspect Dockerfile"

RUN   apt-get update && \
   apt-get -y upgrade && \
   apt-get clean && \
   apt-get install -y python3 python3-pip postgresql-client && \
   rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install \
	--no-cache \
	--disable-pip-version-check \
	--requirement requirements.txt && \
    rm -rf /.cache/pip


EXPOSE 8081
ENTRYPOINT ["bash"]

