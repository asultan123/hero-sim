add_library(xilinx-modules "")
target_include_directories(xilinx-modules PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(xilinx-modules PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/demo-dma.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xilinx-axidma.cc"
)

target_link_libraries(xilinx-modules PkgConfig::SYSTEMC PkgConfig::TLM2)