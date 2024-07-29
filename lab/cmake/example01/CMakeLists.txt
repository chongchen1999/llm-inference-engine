cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(CALC)

set(src_list add.cpp;div.cpp;main.cpp;mul.cpp;sub.cpp)
add_executable(app  ${src_list})
