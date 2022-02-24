cmake_minimum_required(VERSION 3.13)

project(Processor
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS_DEBUG "-pg -DMEM_WAVE_TRACE")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(TLM2 REQUIRED IMPORTED_TARGET tlm)

# add_subdirectory(xilinx)

add_library(cnn_processor INTERFACE)
target_include_directories(cnn_processor INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(cnn_processor INTERFACE Boost::program_options PkgConfig::SYSTEMC PkgConfig::TLM2 xtensor xtensor-blas xtensor::optimize xtensor::use_xsimd)

add_executable(hero_sim_backend "")
target_sources(hero_sim_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/hero_sim_backend.cc")
target_link_libraries(hero_sim_backend cnn_processor)

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
