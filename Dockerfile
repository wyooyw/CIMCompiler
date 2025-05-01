# 基础镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app/CIMCompiler


# 安装必要的软件
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    wget \
    zip

# Install Cmake 3.26.0-rc5
RUN mkdir -p /app/libs/CMake
RUN cd /app/libs/CMake && \
    wget https://cmake.org/files/v3.26/cmake-3.26.0-rc5-linux-x86_64.tar.gz && \
    tar -zxvf cmake-3.26.0-rc5-linux-x86_64.tar.gz cmake-3.26.0-rc5-linux-x86_64
ENV PATH="/app/libs/CMake/cmake-3.26.0-rc5-linux-x86_64/bin:${PATH}"

# Install Ninja 1.8.2
RUN mkdir -p /app/libs/Ninja
RUN cd /app/libs/Ninja && \
    wget https://github.com/ninja-build/ninja/archive/refs/tags/v1.8.2.tar.gz && \
    tar -zxvf v1.8.2.tar.gz ninja-1.8.2 && \
    cd ninja-1.8.2 && \
    python3 ./configure.py --bootstrap
ENV PATH="/app/libs/Ninja/ninja-1.8.2:${PATH}"

# g++ 11.4.0; gcc 11.4.0; make 4.3
RUN g++ --version && \
    gcc --version && \
    make --version

# ccache 3.4.1
RUN mkdir -p /app/libs/CCache
RUN cd /app/libs/CCache && \
    wget https://github.com/ccache/ccache/releases/download/v3.4.1/ccache-3.4.1.tar.gz && \
    tar -zxvf ccache-3.4.1.tar.gz && \
    cd ccache-3.4.1 && \
    ./configure && \
    make && \
    make install

# llvm
RUN apt install -y llvm-12
ENV PATH="/usr/lib/llvm-12/bin:${PATH}"

# other packages
RUN apt install -y clang-12 lld-12 openjdk-17-jdk libgflags-dev libunwind-dev

# ld.ldd
# RUN apt install -y 

# # jdk
# RUN apt install -y 

# RUN apt install 

# python packages
RUN pip3 install bitarray==3.3.1 \
    ipdb==0.13.13 \
    Jinja2==3.1.6 \
    numpy==2.2.5 \
    pandas==2.2.3 \
    plotly==6.0.1 \
    pytest==8.3.5 \
    streamlit==1.44.1 \
    tqdm==4.67.1 \
    pytest-xdist \
    && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# boost
RUN apt install -y libboost-all-dev
RUN apt install -y vim

# for vscode
RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd && \
    echo 'root:yourpassword' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session required pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    /usr/sbin/sshd

# islpy
RUN pip install islpy==2023.2.5 sympy==1.11.1
