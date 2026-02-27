FROM slam-baselines/ros-base:noetic

# LIO-SAM dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-navigation \
    ros-noetic-robot-localization \
    ros-noetic-robot-state-publisher \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# GTSAM 4.x (required by LIO-SAM)
RUN add-apt-repository ppa:borglab/gtsam-release-4.1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgtsam-dev libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

# LIO-SAM
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/TixiaoShan/LIO-SAM.git

# Patch deprecated opencv/cv.h → opencv2/opencv.hpp for OpenCV 4
RUN sed -i 's|#include <opencv/cv.h>|#include <opencv2/opencv.hpp>|g' \
    /catkin_ws/src/LIO-SAM/include/utility.h

# Fix PCL 1.10 + FLANN serialization bug (known Ubuntu 20.04 issue)
RUN sed -i '34s|.*|// &|' /usr/include/flann/util/serialization.h || true

# Ensure LIO-SAM builds with C++17
RUN sed -i '/project(lio_sam)/a set(CMAKE_CXX_STANDARD 17)' \
    /catkin_ws/src/LIO-SAM/CMakeLists.txt

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -j$(nproc)"

# Copy scripts and config
COPY common/kitti_to_rosbag.py /scripts/kitti_to_rosbag.py
COPY common/extract_poses.py /scripts/extract_poses.py
COPY lio_sam/config/ /config/
COPY lio_sam/launch/ /launch/
COPY lio_sam/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

ENTRYPOINT ["/scripts/run.sh"]
