lio_sam:

  # Topics — match our rosbag topic names
  pointCloudTopic: "/velodyne_points"
  imuTopic: "/imu/data"
  odomTopic: "odometry/imu"
  gpsTopic: "odometry/gpsz"

  # Frames
  lidarFrame: "velodyne"
  baselinkFrame: "velodyne"
  odometryFrame: "odom"
  mapFrame: "map"

  # GPS (disabled for KITTI Odometry).
  # useImuHeadingInitialization = false: OxTS yaw is an absolute ENU heading
  # (north = 0, CCW), so LIO-SAM would otherwise rotate its first pose by
  # that absolute angle, leaving the trajectory offset from KITTI GT's
  # "first pose = identity" convention by the vehicle's initial heading
  # (~60 deg on Seq 00). Empirically this flag is also necessary to avoid
  # IMU-preintegration runaway under nav-frame accel + datasheet noise;
  # see SUP-01 P0-1 rework notes in pipeline-notes for measurements.
  useImuHeadingInitialization: false
  useGpsElevation: false
  gpsCovThreshold: 2.0
  poseCovThreshold: 25.0

  # Save
  savePCD: false
  savePCDDirectory: "/tmp/lio_sam/"

  # Sensor: KITTI Velodyne HDL-64E
  sensor: velodyne
  N_SCAN: 64
  Horizon_SCAN: 1800
  downsampleRate: 1
  lidarMinRange: 5.0
  lidarMaxRange: 100.0

  # IMU Settings — OxTS RT3003 datasheet values. R1 loosening experiment
  # (100× σ) made APE 6×worse (1076m → 6406m), so the "Large velocity, reset
  # IMU-preintegration!" at 470+/4541 frames is NOT a noise tuning issue —
  # it's structural (IMU content vs LIO-SAM expectations). Keep datasheet
  # values; root cause tracked as SUP-01 P0-3 in pipeline-notes §20.
  imuAccNoise: 3.9939570888238808e-03
  imuGyrNoise: 1.5636343949698187e-03
  imuAccBiasN: 6.4356659353532566e-05
  imuGyrBiasN: 3.5640318696367613e-05
  imuGravity: 9.80511
  imuRPYWeight: 0.01

  # Extrinsics — RENDERED AT RUNTIME by envsubst (Stage C of SUP-01 P0-1).
  # run.sh derives R/T from KITTI Raw calib_imu_to_velo.txt for the current
  # sequence date and sets KITTI_EXT_TRANS / KITTI_EXT_ROT before launching.
  # A single image slam-baselines/lio_sam:latest serves all sequences; the
  # per-sequence calibration is picked up from the $SEQ env var. Runtime
  # rendering replaces the build-time `inject_kitti_extrinsic.py` pipeline,
  # which suffered from a docker layer-cache divergence (see §20.4 in
  # refs/pipeline-notes.md). Do NOT hand-edit the rendered file — it is
  # regenerated every container start.
  # extrinsicRPY is set equal to extrinsicRot; forcing identity regresses
  # Seq 00 SE(3) APE by ~15 m because LIO-SAM treats RPY as exact.
  extrinsicTrans: [${KITTI_EXT_TRANS}]
  extrinsicRot: [${KITTI_EXT_ROT}]
  extrinsicRPY: [${KITTI_EXT_ROT}]

  # LOAM feature threshold
  edgeThreshold: 1.0
  surfThreshold: 0.1
  edgeFeatureMinValidNum: 10
  surfFeatureMinValidNum: 100

  # Voxel filter
  odometrySurfLeafSize: 0.4
  mappingCornerLeafSize: 0.2
  mappingSurfLeafSize: 0.4

  # Motion constraint
  z_tollerance: 1000
  rotation_tollerance: 1000

  # CPU
  numberOfCores: 4
  mappingProcessInterval: 0.15

  # Surrounding map
  surroundingkeyframeAddingDistThreshold: 1.0
  surroundingkeyframeAddingAngleThreshold: 0.2
  surroundingKeyframeDensity: 2.0
  surroundingKeyframeSearchRadius: 50.0

  # Loop closure
  loopClosureEnableFlag: true
  loopClosureFrequency: 1.0
  surroundingKeyframeSize: 50
  historyKeyframeSearchRadius: 15.0
  historyKeyframeSearchTimeDiff: 30.0
  historyKeyframeSearchNum: 25
  historyKeyframeFitnessScore: 0.3

  # Visualization
  globalMapVisualizationSearchRadius: 1000.0
  globalMapVisualizationPoseDensity: 10.0
  globalMapVisualizationLeafSize: 1.0
