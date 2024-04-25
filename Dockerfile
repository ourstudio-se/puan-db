FROM --platform=linux/amd64 ubuntu:20.04

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Install additional dependencies
RUN apt-get update && \
    apt-get install -y python3-dev libblas-dev liblapack-dev libatlas-base-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# Create a conda environment and install Python and cvxopt
RUN conda create -n myenv python=3.9 -y 
RUN conda conda init & activate myenv
RUN conda install -c conda-forge cvxopt

# Activate the conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate myenv" >> ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

WORKDIR /code

ENV APP_PORT=50051

COPY ./requirements.txt /code/requirements.txt 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code

CMD ["python", "server.py"]