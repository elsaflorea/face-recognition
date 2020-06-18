FROM ubuntu:latest

RUN apt-get update -y \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive \
       apt-get install -y \
            build-essential \
            cmake \
            unzip \
            wget \
            make \
            pkg-config \
            libjpeg-dev \
            libpng-dev \
            libtiff-dev \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev \
            libv4l-dev \
            libxvidcore-dev \
            libx264-dev \
            libgtk-3-dev \
            libatlas-base-dev \
            gfortran \
            python3-dev \
    && mkdir -p /home/root/workspace

WORKDIR /home/root/workspace

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.3.0.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.3.0.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-4.3.0 opencv \
    && mv opencv_contrib-4.3.0 opencv_contrib \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm -rf get-pip.py /home/root/.cache/pip \
    && pip install numpy \
    && mkdir -p /home/root/workspace/opencv/build

WORKDIR /home/root/workspace/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/home/root/workspace/opencv_contrib/modules \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D BUILD_TESTS=OFF \
        -D WITH_EIGEN=OFF \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_VTK=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..
RUN /bin/bash -c "make"
RUN /bin/bash -c "make install"
RUN /bin/bash -c "ldconfig"
RUN apt update -y








