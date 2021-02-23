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

enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
