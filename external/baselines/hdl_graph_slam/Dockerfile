FROM slam-baselines/ros-base:noetic

# hdl_graph_slam dependencies
# g2o is installed via ROS package (ros-noetic-libg2o) on Focal
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-geodesy \
    ros-noetic-nmea-msgs \
    ros-noetic-interactive-markers \
    ros-noetic-libg2o \
    ros-noetic-tf-conversions \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# ndt_omp (required by hdl_graph_slam)
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/koide3/ndt_omp.git

# fast_gicp
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/SMRT-AIST/fast_gicp.git

# hdl_graph_slam
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/koide3/hdl_graph_slam.git

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -j$(nproc)"

# Copy scripts and config
COPY common/kitti_to_rosbag.py /scripts/kitti_to_rosbag.py
COPY common/extract_poses.py /scripts/extract_poses.py
COPY hdl_graph_slam/config/ /config/
COPY hdl_graph_slam/launch/ /launch/
COPY hdl_graph_slam/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

ENTRYPOINT ["/scripts/run.sh"]
