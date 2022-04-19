
function(compile target)
add_executable(${target} "")
target_sources(${target}
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cc"
  )
target_link_libraries(${target} cnn_processor)
endfunction(compile)

function(do_test target result)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

compile(Memory_tb)
compile(AddressGenerator_tb)
compile(SAM_tb)
compile(SSM_tb)

do_test(SAM_tb "ALL TESTS PASS")
do_test(SSM_tb "ALL TESTS PASS")
do_test(AddressGenerator_tb "ALL TESTS PASS")
do_test(Memory_tb "ALL TESTS PASS")
do_test(hero_sim_backend "ALL TESTS PASS")
