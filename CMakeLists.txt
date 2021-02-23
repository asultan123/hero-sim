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

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")

add_executable(sock2sig_tb "")
target_include_directories(sock2sig_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(sock2sig_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bytePrinter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sock2sig.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stringProducer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/sock2sig_tb.cc"
)

target_link_libraries(sock2sig_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(sock2sig_tb
    PUBLIC -Wall
)

# add_executable(AddressGenerator_tb "")
# target_include_directories(AddressGenerator_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# target_sources(AddressGenerator_tb
#     PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator_tb.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_Channel.cc"
# )


# target_link_libraries(AddressGenerator_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

# target_compile_options(AddressGenerator_tb
#     PUBLIC -Wall
# )


# add_executable(Memory_tb "")
# target_include_directories(Memory_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# target_sources(Memory_tb
#     PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/AddressGenerator.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_tb.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/Memory_Channel.cc"
# )


# target_link_libraries(Memory_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

# target_compile_options(Memory_tb
#     PUBLIC -Wall
# )


# add_executable(Connector_tb "")
# target_include_directories(Connector_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# target_sources(Connector_tb
#     PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalControl.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/Connector_tb.cc"
# )


# target_link_libraries(Connector_tb PkgConfig::SYSTEMC PkgConfig::TLM2)

# target_compile_options(Connector_tb
#     PUBLIC -Wall
# )
