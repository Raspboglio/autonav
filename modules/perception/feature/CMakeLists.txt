
find_package(OpenCV REQUIRED)


##########################
######### feature ########
##########################
add_library(feature
src/ORBFeatureExtractor.cpp include/feature/ORBFeatureExtractor.hpp)

target_link_libraries(feature
${OpenCV_LIBS})

if(USE_VPI)
    find_package(vpi REQUIRED)
    target_compile_definitions(feature PUBLIC USE_VPI)
    # set(USE_VPI ON)
    message(STATUS "Found VPI: Version ${vpi_VERSION}")
    target_link_libraries(feature vpi)
endif()
target_include_directories(feature
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

add_library(autonav::feature ALIAS feature)