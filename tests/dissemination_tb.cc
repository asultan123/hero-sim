/**
 * @file dissemination_tb.cc
 * @author Vincent Zhao
 * @brief Basic test of program dissemination on a simple memory layout.
 *
 * Sends a set of distinct programs to a set of connected memories with address generators. The
 * layout of the memory is one parent SAM with 3 children SAMs attached.
 */

#include "dissemination_tb.hh"

#include <cstddef>
#include <iostream>
#include <ostream>

#include "AddressGenerator.hh"
#include "Memory_Channel.hh"
#include "SAM.hh"
#include "dmaTestRegisterMap.hh"

namespace {
constexpr unsigned int dutMemLength = 128;
constexpr unsigned int dutMemWidth = 1;
constexpr unsigned int dutMemChannelCount = 4;
constexpr size_t samCount = 4;
}  // namespace

DISSEMINATION_TB::DISSEMINATION_TB(sc_module_name moduleName)
    : sc_module(moduleName),
      tf(sc_create_vcd_trace_file("ProgTrace")),
      control("global-control-channel", sc_time(1, SC_NS), tf),
      sams("sams", samCount,
           SAMCreator<sc_int<32>>(control, dutMemChannelCount, dutMemLength, dutMemWidth, tf, 0,
                                  samCount - 1)),
      dmaCompleteSigs("dma-complete-sigs", dutMemChannelCount),
      externalChannelReadBus("ext-channel-read-bus", dutMemChannelCount * samCount),
      externalChannelWriteBus("ext-channel-write-bus", dutMemChannelCount * samCount),
      dmaDataReadyValidBus("dma-data-ready-valid-bus", dutMemChannelCount * samCount),
      dmaPeriphReadyValidBus("dma-assert-read-bus", dutMemChannelCount),
      dmaTLastBus("dma-t-last-bus", dutMemChannelCount),
      programSigs("program-sigs", 2),
      dummySigs("dummy-sigs", 3),
      channelEnable(control, tf),
      programSender(control, tf) {
  externalChannelWriteBus[0] =
      0xDEADBEEF;  // Set a recognizable value to detect shift to program data

  // Bindings
  channelEnable.outputValid.bind(dmaDataReadyValidBus[0]);
  channelEnable.tLast.bind(dmaTLastBus[0]);
  channelEnable.inputReady.bind(dmaDataReadyValidBus[1]);

  for (size_t ii = 0; ii < sams.size(); ii++) {
    SAM<sc_int<32>>& sam = sams[ii];
    for (size_t jj = 0; jj < sam.read_channel_data.size(); jj++) {
      sam.read_channel_data[jj][0].bind(
          externalChannelReadBus[jj + ii * sam.read_channel_data.size()]);
      sam.write_channel_data[jj][0].bind(
          externalChannelWriteBus[jj + ii * sam.read_channel_data.size()]);
    }
  }

  programSender.output.bind(programSigs[0]);
  sams[0].program_in.bind(programSigs[0]);
  sams[0].program_out.bind(programSigs[1]);
  for (short ii = 1; ii < 4; ii++) {
    sams[ii].program_in.bind(programSigs[1]);
    sams[ii].program_out.bind(dummySigs[ii - 1]);
  }
}

int DISSEMINATION_TB::runTB() {
  bool success;
  success = disseminatePrograms();

  if (success) {
    std::cout << "TEST SUCCEEDED";
  } else {
    std::cout << "TEST FAILED";
  }

  std::cout << std::endl;

  return 0;
}

bool DISSEMINATION_TB::disseminatePrograms() {
  std::cout << "Testing program dissemination..." << std::endl;

  // Reset sequence
  control.set_reset(true);
  control.set_program(false);

  sc_start(2, sc_core::SC_NS);

  control.set_reset(false);

  // Prepare programs
  vector<Descriptor_2D> tempProgram;

  for (size_t ii = 0; ii < samCount; ii++) {
    tempProgram.clear();
    // Descriptors have IDs encoded in descriptors to identify that program was disseminated to
    // correct module
    Descriptor_2D waitDescriptor(1, ii, DescriptorState::WAIT, ii, ii, 0, 0);
    Descriptor_2D suspendDescriptor(2, ii, DescriptorState::SUSPENDED, 0, 0, 0, 0);
    tempProgram.push_back(waitDescriptor);
    tempProgram.push_back(suspendDescriptor);
    programSender.addProgram(tempProgram);

    for (short jj = 0; jj < 4; jj++) {
      if (jj == 0)
        sams[ii].channels[jj].set_mode(MemoryChannelMode::READ);
      else
        sams[ii].channels[jj].set_mode(MemoryChannelMode::WRITE);
    }
  }

  control.set_program(true);
  std::cout << "Load programs..." << std::endl;
  sc_start(100, SC_NS);
  control.set_program(false);

  std::cout << "Program dissemination complete" << std::endl;

  return true;
}

int sc_main(int argc, char* argv[]) {
  DISSEMINATION_TB dmaTB;

  return dmaTB.runTB();
}