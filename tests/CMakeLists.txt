add_executable(sock2sig_tb "")
target_include_directories(sock2sig_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_sources(sock2sig_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sock2sig_tb.cc"
)

target_link_libraries(sock2sig_tb cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(sock2sig_tb
    PUBLIC -Wall
)

add_executable(SAM_tb "")
target_include_directories(SAM_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_sources(SAM_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/SAM_tb.cc"
)

target_link_libraries(SAM_tb cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(SAM_tb
    PUBLIC -Wall
)

add_executable(AddressGenerator_tb "")
target_include_directories(AddressGenerator_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_sources(AddressGenerator_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/AddressGenerator_tb.cc"
)


target_link_libraries(AddressGenerator_tb cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(AddressGenerator_tb
    PUBLIC -Wall
)


add_executable(Memory_tb "")
target_include_directories(Memory_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_sources(Memory_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Memory_tb.cc"
)


target_link_libraries(Memory_tb cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(Memory_tb
    PUBLIC -Wall
)


add_executable(Connector_tb "")
target_include_directories(Connector_tb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_sources(Connector_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Connector_tb.cc"
)


target_link_libraries(Connector_tb cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(Connector_tb
    PUBLIC -Wall
)

add_executable(axidma_mem_tb "")
target_sources(axidma_mem_tb
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/axidma_mem_tb.cc"
)

target_link_libraries(axidma_mem_tb cnn_processor xilinx-modules PkgConfig::SYSTEMC PkgConfig::TLM2)

target_compile_options(axidma_mem_tb
    PUBLIC -Wall
)

function(do_test target result)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

do_test(SAM_tb "ALL TESTS PASS")
do_test(AddressGenerator_tb "ALL TESTS PASS")
do_test(Connector_tb "ALL TESTS PASS")
do_test(Memory_tb "ALL TESTS PASS")
do_test(sock2sig_tb "ALL TESTS PASS")