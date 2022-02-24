function(compile_test target)
  add_executable(${target} "")
  target_sources(${target}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cc"
    )
  target_link_libraries(${target} cnn_processor PkgConfig::SYSTEMC PkgConfig::TLM2)
  target_compile_options(${target}
    PUBLIC -Wall
  )
endfunction(compile_test)

function(do_test target result)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

compile_test(hero_sim_backend)
compile_test(poly_compute_tb)
compile_test(Connector_tb)
compile_test(Memory_tb)
compile_test(AddressGenerator_tb)
compile_test(SAM_tb)

do_test(SAM_tb "ALL TESTS PASS")
do_test(AddressGenerator_tb "ALL TESTS PASS")
do_test(Connector_tb "ALL TESTS PASS")
do_test(Memory_tb "ALL TESTS PASS")
do_test(sock2sig_tb "ALL TESTS PASS")
do_test(poly_compute_tb "ALL TESTS PASS")
do_test(hero_sim_backend "ALL TESTS PASS")