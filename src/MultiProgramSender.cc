/**
 * @file MultiProgramSender.cc
 * @author Vincent Zhao
 * @brief Implementation of module to serialize and clock out multiple address generator programs.
 */

#include "MultiProgramSender.hh"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "AddressGenerator.hh"
#include "GlobalControl.hh"

template <typename DataType>
MultiProgramSender<DataType>::MultiProgramSender(GlobalControlChannel_IF& control,
                                                 sc_trace_file* tf, uint16_t startUid,
                                                 sc_module_name moduleName)
    : sc_module(moduleName), currentUid(startUid) {
  SC_THREAD(sendPrograms);
  clk.bind(control.clk());
  sc_trace(tf, output, output.name());
  DataType test;
  assert(test.length() % 8 == 0);
  dataLengthBytes = test.length() / 8;
}

template <typename DataType>
void MultiProgramSender<DataType>::addProgram(std::vector<Descriptor_2D>& program) {
  Program_Hdr header = {.uid = currentUid++, .num_descriptors = program.size()};
  addToStream(header);
  for (Descriptor_2D& descriptor : program) {
    addToStream(descriptor);
  }
}

template <typename DataType>
void MultiProgramSender<DataType>::sendPrograms() {
  // Wait for reset sequence
  wait(2, SC_NS);

  // Add padding as needed
  size_t remainder = bytestream.size() % dataLengthBytes;
  if (remainder) bytestream.resize(bytestream.size() + dataLengthBytes - remainder);

  uint8_t* addr = bytestream.data();
  DataType outputVal;
  if (outputVal.length() % 8) throw std::runtime_error("Width is not byte-aligned");
  while (addr < bytestream.data() + bytestream.size()) {
    uint64_t value = 0;
    memcpy(&value, addr, outputVal.length() / 8);
    outputVal = value;
    output = outputVal;
    addr += outputVal.length() / 8;
    wait(clk->negedge_event());
  }

  std::cout << "Programs sent" << std::endl;
}

template <typename DataType>
template <typename T>
void MultiProgramSender<DataType>::addToStream(T& item) {
  uint8_t* addr = reinterpret_cast<uint8_t*>(&item);
  for (size_t ii = 0; ii < sizeof(T); ii++) {
    bytestream.push_back(*addr);
    addr++;
  }
}

template class MultiProgramSender<sc_int<32>>;