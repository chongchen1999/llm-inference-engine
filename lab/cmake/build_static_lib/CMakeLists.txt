cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(make_static_library)

include_directories(${PROJECT_SOURCE_DIR}/includes)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB src_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(calc STATIC ${src_files})
