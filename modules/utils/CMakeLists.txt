find_package(OpenCV REQUIRED)

##########################
#### perception_utils ####
##########################

add_library(perception_utils
include/utils/StereoImage.hpp)

set_target_properties(perception_utils PROPERTIES
    LINKER_LANGUAGE CXX)

target_link_libraries(perception_utils ${OpenCV_LIBS})
target_include_directories(perception_utils PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    include)

add_library(autonav::perception_utils ALIAS perception_utils)