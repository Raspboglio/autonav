# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

unset(COLMAP_DOWNLOAD_ENABLED)
unset(_MSC_VER)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(FETCH_POSELIB ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

set(GPU_ENABLED ON)
set(CUDA_ENABLED ON)
add_definitions("-DCOLMAP_GPU_ENABLED")


if(NOT DEFINED COLMAP_DOWNLOAD_ENABLED)
    message("WARNING: COLMAP_DOWNLOAD_ENABLED is not defined. Disabling download support.")
endif()

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    set(_fetch_content_declare_args "SYSTEM" "EXCLUDE_FROM_ALL")
endif()
FetchContent_Declare(poselib
    URL https://github.com/PoseLib/PoseLib/archive/8028473d92c9347794a0e3d3541863b5cbb15743.zip
    URL_HASH SHA256=20ff75d14c839dcab3d38fb51503c47088898f394b812472a758ec17720a54c2
    ${_fetch_content_declare_args}
)
message(STATUS "Configuring PoseLib...")
set(MARCH_NATIVE OFF CACHE BOOL "")
FetchContent_MakeAvailable(poselib)
message(STATUS "Configuring PoseLib... done")

include_directories(
    ../
)

if(IS_MSVC)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
# Avoid pulling in too many header files through <windows.h>
add_definitions("-DWIN32_LEAN_AND_MEAN")
elseif(IS_GNU OR IS_CLANG)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()


set(COLMAP_FIND_QUIETLY FALSE)
include(cmake/FindDependencies.cmake)

if(CUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

    # Suppress warnings:
    # ptxas warning : Stack size for entry function X cannot be statically determined.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")
endif()

add_subdirectory(thirdparty)
add_subdirectory(util)
add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(math)
add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(scene)
add_subdirectory(sensor)
add_subdirectory(sfm)
add_subdirectory(tools)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif()
