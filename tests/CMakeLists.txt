cmake_minimum_required(VERSION 2.8.3)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find all main*.cpp files and store in list mains
file(GLOB_RECURSE tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/test*.cpp")
foreach(test_file IN LISTS tests)
    # Get file name without directory
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES} supervio_nodelet)
    #gtest_add_tests(${test_name})

endforeach()