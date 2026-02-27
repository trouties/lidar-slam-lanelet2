FROM slam-baselines/ros-base:noetic

# FAST-LIO2 dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-cv-bridge \
    ros-noetic-eigen-conversions \
    ros-noetic-tf-conversions \
    && rm -rf /var/lib/apt/lists/*

# livox_ros_driver (build dependency, even though we use Velodyne input)
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver.git

# FAST-LIO2
RUN cd /catkin_ws/src && \
    git clone --depth 1 --recursive https://github.com/hku-mars/FAST_LIO.git

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -j$(nproc)"

# Copy scripts and config
COPY common/kitti_to_rosbag.py /scripts/kitti_to_rosbag.py
COPY common/extract_poses.py /scripts/extract_poses.py
COPY fast_lio2/config/ /config/
COPY fast_lio2/launch/ /launch/
COPY fast_lio2/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

ENTRYPOINT ["/scripts/run.sh"]
