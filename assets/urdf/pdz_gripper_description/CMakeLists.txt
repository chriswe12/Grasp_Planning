cmake_minimum_required(VERSION 3.8)
project(pdz_gripper_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES README.md D405_FRAME_REFERENCE.md GEOMETRY_STATISTICS.md
        COORDINATE_FRAME_AUDIT.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
