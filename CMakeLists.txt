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

add_executable(SAM_tb "")
target_include_directories(SAM_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# target_sources(SAM_tb
#     PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/bytePrinter.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/sock2sig.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/stringProducer.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/tests/SAM_tb.cc"
# )
target_sources(SAM_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Connector.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_Channel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SAM.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SAM_tb.cc"
)

target_link_libraries(SAM_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(SAM_tb
    PUBLIC -Wall
)

