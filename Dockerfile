FROM ros:melodic as melodic-cuda

RUN apt-get update && apt-get install wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
    apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install cuda cuda-drivers && \
    rm cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
    apt-get install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
    apt-get update && \
    apt-get install libcudnn7 libcudnn7-dev && \
    rm nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

FROM melodic-cuda
RUN apt-get update && \
    apt-get install -y gcc-7 python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge git python3-pip && \
    pip3 install rospkg

RUN mkdir ros &&\
    cd ros &&\
    catkin init &&\
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so &&\
    catkin config --install &&\
    git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv &&\
    cd src/vision_opencv/ &&\
    git checkout melodic &&\
    cd ../../ &&\
    git clone --recursive https://github.com/Tacha-S/smap_ros.git src/smap_ros && \
    pip3 install -U pip && \
    pip3 install torch torchvision numpy==1.19.5 && \
    pip3 install -r src/smap_ros/SMAP/requirements.txt && \
    /bin/bash -c "source /opt/ros/melodic/setup.bash;catkin build" && \
    echo "source /ros/devel/setup.bash" >> /root/.bashrc && \
    echo "source /ros/install/setup.bash --extend" >> /root/.bashrc

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN cd /ros/src/smap_ros/SMAP/extensions && \
    python3 setup.py install
