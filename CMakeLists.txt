# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (flatctr)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/cxxopts)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/fast_float)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/libcuckoo)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/deps/spdlog)

set(SOURCES src/flatctr.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/include/)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cxxopts
        fast_float
        libcuckoo
        spdlog::spdlog
)
