cmake_minimum_required(VERSION 3.13)

project(HeroSim-Backend
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(PkgConfig REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(fmt REQUIRED)

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)

add_subdirectory(hero-sim-proto)


add_library(cnn_processor INTERFACE)
target_include_directories(cnn_processor INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


target_link_libraries(cnn_processor INTERFACE PkgConfig::SYSTEMC fmt::fmt-header-only Boost::program_options xtensor xtensor-blas xtensor::optimize xtensor::use_xsimd)

add_executable(hero_sim_backend "")
target_sources(hero_sim_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/hero_sim_backend.cc")
target_link_libraries(hero_sim_backend cnn_processor proto)

install(TARGETS hero_sim_backend
        CONFIGURATIONS Release
        RUNTIME DESTINATION /bin)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "SULTAN") 
INCLUDE(CPack)

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
