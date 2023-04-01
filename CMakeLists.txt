# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (FlatCTR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Create a sources variable with a link to all cpp files to compile
set(SOURCES
        src/main.cpp
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/cxxopts)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/fast_float)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/libcuckoo)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/spdlog)

# Add an executable with the above sources
add_executable(flatctr ${SOURCES})

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(flatctr
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/src/include/
)

target_link_libraries(flatctr
    PRIVATE
        cxxopts
        fast_float
        libcuckoo
        spdlog::spdlog
)
