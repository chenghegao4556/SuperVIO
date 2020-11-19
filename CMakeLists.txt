cmake_minimum_required(VERSION 2.8.3)
project(supervio)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Ofast")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(Torch_DIR /home/chenghe/3rdparty/pytorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
FIND_PACKAGE(DBoW3 REQUIRED )
find_package(catkin REQUIRED COMPONENTS
        roscpp roslib pcl_ros sensor_msgs cv_bridge image_transport tf_conversions
        nodelet camera_info_manager
)
catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp nodelet image_transport sensor_msgs camera_info_manager
        LIBRARIES ${PROJECT_NAME}_nodelet
)

find_package(OpenCV 3.4.1)
find_package(Pangolin REQUIRED)
add_definitions(-DOPENCV_WITH_CUDA)
add_definitions(-DOPENCV_WITH_CONTRIB)
add_definitions(-DROS_ASSERT_ENABLED)

find_package(PkgConfig)

add_library(${PROJECT_NAME}_nodelet
        src/utility/matrix_operation.cpp
        src/imu/imu_states_measurements.cpp
        src/imu/imu_integrator.cpp
        src/imu/imu_checker.cpp
        src/imu/imu_pre_integrator.cpp
        src/imu/imu_noise.cpp
        src/imu/imu_processor.cpp
        src/vision/camera.cpp
        src/vision/feature_extractor.cpp
        src/vision/feature_matcher.cpp
        src/vision/feature_tracker.cpp
        src/vision/triangulator.cpp
        src/vision/vision_measurements.cpp
        src/sfm/initial_sfm.cpp
        src/sfm/absolute_pose.cpp
        src/sfm/relative_pose.cpp
        src/sfm/match.cpp
        src/optimization/factors/imu_factor.cpp
        src/optimization/factors/inverse_depth_factor.cpp
        src/optimization/factors/marginalization_factor.cpp
        src/optimization/factors/cost_function_factory.cpp
        src/optimization/factors/relative_pose_factor.cpp
        src/optimization/parameter_blocks/base_parameter_block.cpp
        src/optimization/parameter_blocks/pose_parameter_block.cpp
        src/optimization/parameter_blocks/inverse_depth_parameter_block.cpp
        src/optimization/parameter_blocks/speed_bias_parameter_block.cpp
        src/optimization/parameter_blocks/parameter_block_factory.cpp
        src/optimization/parameter_blocks/pose_local_parameterization.cpp
        src/optimization/residual_blocks/base_residual_block.cpp
        src/optimization/residual_blocks/pre_integration_residual_block.cpp
        src/optimization/residual_blocks/reprojection_residual_block.cpp
        src/optimization/residual_blocks/marginalization_residual_block.cpp
        src/optimization/residual_blocks/relative_pose_residual_block.cpp
        src/optimization/residual_blocks/residual_block_factory.cpp
        src/optimization/marginalizer.cpp
        src/optimization/optimizer.cpp
        src/optimization/helper.cpp
        src/estimation/estimation_core.cpp
        src/estimation/state_estimator.cpp
        src/estimation/vio_states_measurements.cpp
        src/estimation/visual_imu_alignment.cpp
        src/estimation/parameters.cpp
        src/visualization/visuailizer.cpp
        src/nodelets/super_vio_nodelet.cpp
        src/dense_mapping/densifier.cpp
        src/dense_mapping/fast_bilateral_solver.cpp
        src/dense_mapping/triangle.cpp
        src/dense_mapping/delaunay.cpp
        src/dense_mapping/internal/draw_function.cpp
        src/dense_mapping/mesh_regularizer.cpp
        src/dense_mapping/depth_interpolator.cpp
        src/loop_closure/image_retrieval.cpp
        src/loop_closure/loop_closure_estimator.cpp
)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${DBoW3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${DBoW3_LIBRARIES}
)
add_dependencies(${PROJECT_NAME}_nodelet
        ${catkin_EXPORTED_TARGETS})

add_executable(super_vio_node src/ros_node.cpp)
target_link_libraries(super_vio_node ${PROJECT_NAME}_nodelet)
add_executable(dense_mapping_node src/dense_mapping_node.cpp)
target_link_libraries(dense_mapping_node ${PROJECT_NAME}_nodelet)
add_executable(ar_depth_node src/ar_depth_node.cpp)
target_link_libraries(ar_depth_node ${PROJECT_NAME}_nodelet)

add_subdirectory(tests)

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

# Install nodelet library
install(TARGETS ${PROJECT_NAME}_nodelet
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Install header files
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install xml files
install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
