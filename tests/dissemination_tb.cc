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
      // sock2sig(control, 256, tf),
      // sig2sock(control, 256, tf),
      dmaCompleteSigs("dma-complete-sigs", dutMemChannelCount),
      // mm2s("axidma-mm2s"),
      // s2mms("s2mms", ),
      // bus("bus"),
      // mem("memory", SC_ZERO_TIME, MEM_SIZE),
      // dmaTester(true),
      externalChannelReadBus("ext-channel-read-bus", dutMemChannelCount * samCount),
      externalChannelWriteBus("ext-channel-write-bus", dutMemChannelCount * samCount),
      dmaDataReadyValidBus("dma-data-ready-valid-bus", dutMemChannelCount * samCount),
      dmaPeriphReadyValidBus("dma-assert-read-bus", dutMemChannelCount),
      dmaTLastBus("dma-t-last-bus", dutMemChannelCount),
      programSigs("program-sigs", 2),
      dummySigs("dummy-sigs", 3),
      channelEnable(control, tf),
      programSender(control, tf) {
  // Bindings
  // bus.memmap(MEM_BASE_ADDR, MEM_SIZE, ADDRMODE_RELATIVE, -1, mem.socket);

  // mm2s bindings
  // bus.memmap(MM2S_BASE_ADDR, MM2S_REG_SIZE, ADDRMODE_RELATIVE, -1, mm2s.tgt_socket);
  // mm2s.init_socket.bind(*bus.t_sk[1]);
  // mm2s.irq.bind(dmaCompleteSigs[0]);

  // sock2sig.inputSock.bind(mm2s.stream_socket);
  // sock2sig.outputSig.bind(externalChannelWriteBus[0]);
  // sock2sig.outputValid.bind(dmaDataReadyValidBus[0]);
  // sock2sig.peripheralReady.bind(sams[0].channel_dma_periph_ready_valid[0]);
  // sock2sig.tLast.bind(dmaTLastBus[0]);
  externalChannelWriteBus[0] = 0xDEADBEEF;

  channelEnable.outputValid.bind(dmaDataReadyValidBus[0]);
  channelEnable.tLast.bind(dmaTLastBus[0]);

  // s2mm bindings
  // bus.memmap(S2MM_BASE_ADDR, MM2S_REG_SIZE, ADDRMODE_RELATIVE, -1, s2mm.tgt_socket);
  // s2mm.init_socket.bind(*bus.t_sk[2]);
  // s2mm.irq.bind(dmaCompleteSigs[1]);

  // sig2sock.outputSock.bind(s2mm.stream_socket);
  // sig2sock.inputSig.bind(externalChannelReadBus[1]);
  // sig2sock.inputReady.bind(dmaDataReadyValidBus[1]);
  // sig2sock.peripheralValid.bind(sam.channel_dma_periph_ready_valid[1]);
  // sig2sock.packetLength.bind(s2mm.packet_length);

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

  // sc_trace(tf, mm2s.irq, "dma-mm2s-ioc");
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