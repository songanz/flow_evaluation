FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
MAINTAINER Subramanya Nageshrao (snageshr@ford.com)

# System
RUN apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y \
	locales \
    vim \
    apt-utils 

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

## ENV
ENV HOME /root
ENV SRC_DIR /usr/local/src
ENV SUMO_1_4 $SRC_DIR/sumo/sumo-1_4_0

## github communications
# ssh stuff
RUN apt update && apt install -yyq ssh && mkdir $HOME/.ssh
COPY id_rsa $HOME/.ssh/id_rsa
COPY id_rsa.pub $HOME/.ssh/id_rsa.pub
RUN chmod 400 $HOME/.ssh/id_rsa && ssh-keyscan github.com >> ~/.ssh/known_hosts

### Pycharm
WORKDIR /tmp
ARG PYCHARM_VERSION=pycharm-community-2021.3.2
ENV PYCHARM_BIN /opt/$PYCHARM_VERSION/bin
RUN wget https://download.jetbrains.com/python/$PYCHARM_VERSION.tar.gz && \
    tar -zxvf $PYCHARM_VERSION.tar.gz -C /opt && \
    rm $PYCHARM_VERSION.tar.gz && \
    ln -s /opt/$PYCHARM_VERSION/bin/pycharm.sh /usr/bin/pycharm.sh

### Anaconda
WORKDIR /tmp
ENV CONDA_BIN /opt/anaconda3/bin
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    /opt/anaconda3/bin/conda create -n flow python=3.8 -y && \
    /opt/anaconda3/bin/conda install -y -n flow pytorch=1.10.0 cpuonly -c pytorch -c anaconda -c nvidia && \
    rm Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    chmod -R 777 /opt/anaconda3/envs/flow
    
##########################################
######## VNC lxde template start #########
##########################################

WORKDIR /
RUN apt update \
    && apt install -y --no-install-recommends software-properties-common curl apache2-utils \
    && apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
    && apt autoclean -y \
    && apt autoremove -y

# install debs error if combine together
RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        xvfb x11vnc \
        vim-tiny firefox ttf-ubuntu-font-family ttf-wqy-zenhei  \
    && apt autoclean -y \
    && apt autoremove -y

RUN apt update \
    && apt install -y gpg-agent \
    && curl -LO https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && (dpkg -i ./google-chrome-stable_current_amd64.deb || apt-get install -fy) \
    && curl -sSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add \
    && rm google-chrome-stable_current_amd64.deb

RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
    && apt autoclean -y \
    && apt autoremove -y


# Additional packages require ~600MB
# libreoffice  pinta language-pack-zh-hant language-pack-gnome-zh-hant firefox-locale-zh-hant libreoffice-l10n-zh-tw

# tini to fix subreap
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# ffmpeg
RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /usr/local/ffmpeg \
    && ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg

RUN rm -vf /var/lib/apt/lists/* && apt update

###### SUMO install

### LIBPROJ for SUMO
WORKDIR /
ENV PATH /opt/anaconda3/envs/flow/bin:$PATH
ENV LIBPROJ_DIR $SRC_DIR/libproj

# Using legacy libproj. Add `apt install libproj-dev` when we update sumo
RUN apt update && apt install -yyq git apt-utils libfox-1.6-dev g++ libgdal-dev libgl2ps-dev libxerces-c-dev libgtest-dev swig default-jdk cmake jq && \
    mkdir $LIBPROJ_DIR && \
    GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa' git clone --depth 1 --branch 5.2.0 git@github.com:OSGeo/PROJ.git -q $LIBPROJ_DIR && \
    cd $LIBPROJ_DIR && \
    ./autogen.sh && ./configure && \
    make -j 24 && \
    make install && \
    make check      

# SUMO versions
RUN GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa' git clone --depth 1 --branch v1_4_0 git@github.com:eclipse/sumo.git -q $SUMO_1_4 && \
    cd $SUMO_1_4 && \
    mkdir build_cmake && \
    cd build_cmake && \
    cmake -DPYTHON_EXECUTABLE:FILEPATH=/opt/anaconda3/envs/flow/bin/python3 .. && \
    make -j 24 && \
    ../bin/sumo  

ENV SUMO_HOME $SUMO_1_4

ENV SUMO_BIN $SUMO_HOME/bin
ENV SUMO_TOOLS $SUMO_HOME/tools
ENV SUMO_BIN $SUMO_HOME/bin
ENV SUMO_TOOLS $SUMO_HOME/tools
ENV PATH $SUMO_BIN:$PATH
ENV PYTHONPATH $SUMO_TOOLS:$PYTHONPATH
ENV CONDA_SSM_BIN /opt/anaconda3/envs/flow/bin

RUN ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

##### Flow
ENV FLOW_HOME $SRC_DIR/flow
WORKDIR $SRC_DIR
RUN GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa' git clone git@github.com:flow-project/flow.git -q
COPY ./requirements.txt $SRC_DIR
RUN cd flow
RUN /bin/bash -c "/bin/sh /etc/profile.d/conda.sh activate flow && pip install --use-deprecated=legacy-resolver -r $SRC_DIR/requirements.txt && pip install jupyter" && \
    chmod -R 777 /opt/anaconda3/envs/flow && \
    rm -rf $SRC_DIR/requirements.txt

# Add startup scripts
COPY ./etc /etc
RUN chmod -R 777 /etc/supervisor/conf.d
RUN chmod -R 777 /etc/lxde
RUN chmod -R 777 /var/log/supervisor

COPY ./usr /usr
RUN chmod -R 777 /usr/local/bin
RUN chmod -R 777 /usr/local/src
RUN chmod -R 777 /var/run

# Set up user env without messing up their ~/.bashrc
RUN printf "if [ \"\$#\" -lt 1 ]; then\n\techo \"Usage: source activate.sh <PATH_TO_FLOW>\"\nelse\n" >> /usr/local/bin/activate.sh && \
    printf "\t. /opt/anaconda3/etc/profile.d/conda.sh\n" >> /etc/bash.bashrc && \
    printf "\texport SUMO_HOME=\${2:-$SUMO_HOME}\n" >> /usr/local/bin/activate.sh && \
    printf "\texport PYTHONPATH=\$FLOW_HOME:\$SUMO_HOME/tools\n" >> /usr/local/bin/activate.sh && \
    printf "\tconda activate flow\n" >> /usr/local/bin/activate.sh && \
    printf "fi\n" >> /usr/local/bin/activate.sh && \
    chmod +x /usr/local/bin/activate.sh && \
    ln -s /usr/local/bin/activate.sh /usr/bin/activate.sh

# Dispose of the annoying geoclue
# RUN apt purge -yyq geoclue-2.0
RUN apt autoremove && sed -i '/geoclue/d' /var/lib/dpkg/statoverride
