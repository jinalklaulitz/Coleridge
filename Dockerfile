#will use nvidia's cuda docker setup
#from ubuntu:18.04
#used some things from the tensorflow dockerfile  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/nvidia.Dockerfile
#from nvidia/cuda:10.0-base-ubuntu18.04
#FROM nvidia/cuda:9.0-base-ubuntu16.04
#trying the pre-build docker container from NVIDIA
#FROM nvcr.io/nvidia/tensorflow:18.11-py3
FROM nvcr.io/nvidia/cuda:9.0-cudnn7.2-devel-ubuntu16.04

MAINTAINER marvin mananghaya <msm796@nyu.edu>

# Run apt to install OS packages
#<original>
#RUN apt update
#RUN apt install -y tree vim curl python3 python3-pip git

#<latest - as of 11/15/18>
#<made use of the dockerfile from the anaconda https://hub.docker.com/r/continuumio/anaconda3/~/dockerfile/>

# declare environmental variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV PATH /opt/conda/bin:$PATH

#install os related stuff
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion gcc g++ python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev vim nano \
# added some lines from the cuda-tensorflow dockerfile
#build-essential \
#cuda-command-line-tools-9-0 \
#cuda-cublas-9-0 \
#cuda-cufft-9-0 \
#cuda-curand-9-0 \
#cuda-cusolver-9-0 \
#cuda-cusparse-9-0 \
#libcudnn7=7.2.1.38-1+cuda9.0 \
#libnccl2=2.2.13-1+cuda9.0 \
#libfreetype6-dev \
#libhdf5-serial-dev \
#libpng12-dev \
#libzmq3-dev \
#pkg-config \
software-properties-common \
unzip \
&& \
apt-get clean && \
rm -rf /var/lib/apt/lists/*


#RUN apt-get update && \
#        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
#        apt-get update && \
#        apt-get install libnvinfer4=4.1.2-1+cuda9.0

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

#install anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


#installs tini (? i think it's garbage collection for containers)
RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

#get environment.yml file
RUN wget --quiet https://raw.githubusercontent.com/jinalklaulitz/Coleridge/master/nlpenv.yml

#create nlp conda environment
RUN conda env create --name nlpEnv --file=nlpenv.yml

#change environment to nlpEnv and install spacy english models (the pip install method doesn't work)
#ENV PATH /opt/conda/envs/nlpEnv/bin:$PATH


#for TF installation
#ARG TF_PACKAGE=tensorflow-gpu

#RUN python -m spacy download en_core_web_sm en_core_web_lg en_vectors_web_lg en_core_web_md
#RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz \
#https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz \
#https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.0.0/en_core_web_md-2.0.0.tar.gz ${TF_PACKAGE}

# Python 3 package install example - <depreciated since we install anaconda>
#RUN pip3 install ipython matplotlib numpy pandas scikit-learn scipy six

#install tensorflow-gpu - old, integrated it with installation of spacy
#ARG TF_PACKAGE=tensorflow-gpu
#RUN ${PIP} install ${TF_PACKAGE}

#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

# create directory for work.
RUN cd /  && mkdir /work

# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition

#LABEL maintainer=jonathan.morgan@nyu.edu
