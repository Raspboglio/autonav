find_package(OpenCV REQUIRED)

add_executable(camera_extraction_test src/CameraExtractionTest.cpp)
target_link_libraries(camera_extraction_test
    stereovision
    perception_utils
    ${OpenCV_LIBS}
)

target_include_directories(camera_extraction_test
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)