FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
ARG spark_uid=185
ARG DEBIAN_FRONTEND=noninteractive

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install java dependencies
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV PATH $PATH:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin:/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin

# Install pyspark and vim, and clean up any unnecessary files
RUN apt-get update && apt-get install -y --no-install-recommends vim \
    && pip install --no-cache-dir pyspark==3.3.2 torch transformers datasets \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Before building the docker image, first either download Apache Spark 3.1+ from
# http://spark.apache.org/downloads.html or build and make a Spark distribution following the
# instructions in http://spark.apache.org/docs/3.1.2/building-spark.html (see
# https://nvidia.github.io/spark-rapids/docs/download.html for other supported versions).  If this
# docker file is being used in the context of building your images from a Spark distribution, the
# docker build command should be invoked from the top level directory of the Spark
# distribution. E.g.: docker build -t spark:3.1.2 -f kubernetes/dockerfiles/spark/Dockerfile .

RUN set -ex && \
    ln -s /lib /lib64 && \
    mkdir -p /opt/spark && \
    mkdir -p /opt/spark/jars && \
    mkdir -p /opt/spark/examples && \
    mkdir -p /opt/spark/work-dir && \
    mkdir -p /opt/sparkRapidsPlugin && \
    touch /opt/spark/RELEASE && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

COPY spark/jars /opt/spark/jars
COPY spark/bin /opt/spark/bin
COPY spark/sbin /opt/spark/sbin
COPY spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt/
COPY spark/examples /opt/spark/examples
COPY spark/kubernetes/tests /opt/spark/tests
COPY spark/data /opt/spark/data

COPY rapids-4-spark_2.12-*.jar /opt/sparkRapidsPlugin
COPY getGpusResources.sh /opt/sparkRapidsPlugin

RUN mkdir /opt/spark/python
# TODO: Investigate running both pip and pip3 via virtualenvs
RUN apt-get update && \
    # apt install -y python python-pip && \
    apt install -y python3 python3-pip && \
    # We remove ensurepip since it adds no functionality since pip is
    # installed on the image and it just takes up 1.6MB on the image
    # rm -r /usr/lib/python*/ensurepip && \
    pip install --upgrade pip setuptools
    # You may install with python3 packages by using pip3.6
    # Removed the .cache to save space
    # rm -r /root/.cache && rm -rf /var/cache/apt/*

COPY spark/python/pyspark /opt/spark/python/pyspark
COPY spark/python/lib /opt/spark/python/lib

ENV SPARK_HOME /opt/spark

WORKDIR /opt/spark/work-dir
RUN chmod g+w /opt/spark/work-dir

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +rx /usr/bin/tini

ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
USER ${spark_uid}
