cmake_minimum_required(VERSION 3.13)

project(Processor
    VERSION 0.1
    DESCRIPTION "Testbench for distributed control based processor SystemC module"
    LANGUAGES CXX
)

set(cmake_cxx_standard 20)
set(cmake_cxx_standard_required on)

set(cmake_build_type debug)

find_package(PkgConfig REQUIRED)

pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(TLM2 REQUIRED IMPORTED_TARGET tlm)

add_executable(processor_tb "")
target_include_directories(processor_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(processor_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bytePrinter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sock2sig.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stringProducer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/processor_tb.cc"
)

target_link_libraries(processor_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(processor_tb
    PUBLIC -Wall
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests)
