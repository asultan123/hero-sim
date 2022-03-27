cmake_minimum_required(VERSION 3.13)

project(Processor
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS_DEBUG "-pg -DMEM_WAVE_TRACE")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(fmt REQUIRED)

# find_program(IWYU_PATH NAMES include-what-you-use iwyu)
# if(NOT IWYU_PATH)
#   message(FATAL_ERROR "Could not find the program include-what-you-use")
# endif()
# set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})
# set(CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
# pkg_check_modules(TLM2 REQUIRED IMPORTED_TARGET tlm)

# add_subdirectory(xilinx)

add_library(cnn_processor INTERFACE)
target_include_directories(cnn_processor INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(cnn_processor INTERFACE PkgConfig::SYSTEMC fmt::fmt-header-only Boost::program_options xtensor xtensor-blas xtensor::optimize xtensor::use_xsimd)

add_executable(hero_sim_backend "")
target_sources(hero_sim_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/hero_sim_backend.cc")
target_link_libraries(hero_sim_backend cnn_processor)

# set_property(TARGET hero_sim_backend PROPERTY CXX_INCLUDE_WHAT_YOU_USE "${iwyu_path};-Xiwyu;--mapping_file=${CMAKE_CURRENT_SOURCE_DIR}/systemc.imp;-Xiwyu;--cxx14ns")

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
