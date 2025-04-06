find_package(OpenCV REQUIRED)

###########
## camera_extraction_test
###########
add_executable(camera_extraction_test src/CameraExtractionTest.cpp)
target_link_libraries(camera_extraction_test
    stereovision
    perception_utils
    autonav::feature
    ${OpenCV_LIBS}
)

target_include_directories(camera_extraction_test
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

###########
## camera_extraction_test
###########
find_package(vpi REQUIRED)
add_executable(vpi_orb_test src/VpiOrbTest.cpp)
target_link_libraries(vpi_orb_test
    vpi
    ${OpenCV_LIBS}
)

target_include_directories(vpi_orb_test
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

