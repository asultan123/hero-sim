cmake_minimum_required(VERSION 3.13)

project(Processor
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-pg -DMEM_WAVE_TRACE")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(TLM2 REQUIRED IMPORTED_TARGET tlm)

add_subdirectory(xilinx)

add_library(cnn_processor "")
target_include_directories(cnn_processor PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_sources(cnn_processor PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bytePrinter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_Channel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SAM.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sock2sig.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stringProducer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProcEngine.cc"
)
target_link_libraries(cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
