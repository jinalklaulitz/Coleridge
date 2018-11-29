from ubuntu:18.04

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
git mercurial subversion gcc g++ python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev vim nano

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
ENV PATH /opt/conda/envs/nlpEnv/bin:$PATH
#RUN python -m spacy download en_core_web_sm en_core_web_lg en_vectors_web_lg en_core_web_md
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz \
https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz \
https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz

# Python 3 package install example - <depreciated since we install anaconda>
#RUN pip3 install ipython matplotlib numpy pandas scikit-learn scipy six

# create directory for work.
RUN cd /  && mkdir /work

# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition


#LABEL maintainer=jonathan.morgan@nyu.edu
