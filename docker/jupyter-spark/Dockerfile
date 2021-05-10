FROM jupyter/base-notebook

USER root

ARG spark_version="3.0.0"
ARG hadoop_version="2.7"
ARG openjdk_version="8"

# Copy the requirements file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/

RUN apt-get update && apt-get install -yq --no-install-recommends \
    # Install all OS dependencies (including Mariadb development files)
    build-essential \
    vim-tiny \
    git \
    libsm6 \
    libxext-dev \
    libxrender1 \
    libmariadb-dev \
    lmodern \
    netcat \
    tzdata \
    unzip \
    nano-tiny \
    # Java JDK installation
    "openjdk-${openjdk_version}-jre-headless" \
    ca-certificates-java \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    # Install Python packages
    && pip install -r /tmp/requirements.txt \
    && conda install -c conda-forge proj==7.2.0 basemap 

# Set the PROJ_LIB environmental variable
ENV PROJ_LIB=/opt/conda/pkgs/proj-7.2.0-h277dcde_2/share/proj 

# Change working directory
WORKDIR /tmp

# Donwload and install Apache Spark
RUN wget https://archive.apache.org/dist/spark/spark-${spark_version}/spark-${spark_version}-bin-hadoop${hadoop_version}.tgz && \
    tar -xzf spark-${spark_version}-bin-hadoop${hadoop_version}.tgz -C /usr/local --owner root && \
    rm spark-${spark_version}-bin-hadoop${hadoop_version}.tgz

WORKDIR /usr/local

# Configure environmental variables for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME:$PATH
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYSPARK_PYTHON=python3
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" 

# Create a symbolic link
RUN ln -s "spark-${spark_version}-bin-hadoop${hadoop_version}" spark

USER $NB_UID

WORKDIR $HOME