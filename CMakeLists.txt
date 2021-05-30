cmake_minimum_required(VERSION 3.13)

project(Processor
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CHECK_INCLUDES ON)

set(CMAKE_BUILD_TYPE DEBUG)

find_package(PkgConfig REQUIRED)

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(TLM2 REQUIRED IMPORTED_TARGET tlm)

find_program(iwyu_path NAMES include-what-you-use iwyu)
if((NOT iwyu_path) AND (DEFINED CHECK_INCLUDES))
    message(FATAL_ERROR "Could not find the program include-what-you-use")
endif()

add_subdirectory(xilinx)

add_library(cnn_processor "")
target_include_directories(cnn_processor PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_sources(cnn_processor PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bytePrinter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/channelEnable.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dmaTester.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_Channel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SAM.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sig2sock.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sock2sig.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stringProducer.cc"
)

target_link_libraries(cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")

if(DEFINED CHECK_INCLUDES)
    set_property(TARGET cnn_processor PROPERTY CXX_INCLUDE_WHAT_YOU_USE "${iwyu_path};-Xiwyu;--mapping_file=${CMAKE_CURRENT_SOURCE_DIR}/systemc.imp;-Xiwyu;--cxx17ns")
endif()