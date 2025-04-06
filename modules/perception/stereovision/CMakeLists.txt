
add_subdirectory(./external)

find_package(OpenCV REQUIRED)

############################
####### stereovision #######
############################

add_library(stereovision
    src/CameraCalibrator.cpp
)

target_include_directories(stereovision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stereovision
    ${OpenCV_LIBS}
    autonav::perception_utils
    colmap_feature
    colmap_controllers
)

add_library(autonav::stereovision ALIAS stereovision)
############################