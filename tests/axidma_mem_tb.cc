#include "axidma_mem_tb.hh"

#include <sysc/communication/sc_signal.h>
#include <sysc/communication/sc_writer_policy.h>
#include <sysc/datatypes/int/sc_int.h>
#include <sysc/kernel/sc_module.h>
#include <sysc/kernel/sc_time.h>
#include <sysc/tracing/sc_trace.h>

#include <cstddef>
#include <iostream>
#include <ostream>

#include "Memory_Channel.hh"
#include "dmaTestRegisterMap.hh"

namespace {
constexpr unsigned int dutMemLength = 128;
constexpr unsigned int dutMemWidth = 1;
constexpr unsigned int dutMemChannelCount = 2;
}  // namespace

DMA_TB::DMA_TB(sc_module_name moduleName)
    : sc_module(moduleName),
      tf(sc_create_vcd_trace_file("ProgTrace")),
      control("global-control-channel", sc_time(1, SC_NS), tf),
      sam("sam", control, dutMemChannelCount, dutMemLength, dutMemWidth, tf),
      sock2sig(control, 256, tf),
      sig2sock(control, 256, tf),
      dmaCompleteSigs("dma-complete-sigs", dutMemChannelCount),
      mm2s("axidma-mm2s"),
      s2mm("axidma-s2mm"),
      bus("bus"),
      mem("memory", SC_ZERO_TIME, MEM_SIZE),
      dmaTester(true),
      externalChannelReadBus("ext-channel-read-bus", dutMemChannelCount),
      externalChannelWriteBus("ext-channel-write-bus", dutMemChannelCount),
      dmaDataReadyValidBus("dma-data-ready-valid-bus", dutMemChannelCount),
      dmaPeriphReadyValidBus("dma-assert-read-bus", dutMemChannelCount),
      channelEnable(control, tf) {
  // Bindings
  bus.memmap(MEM_BASE_ADDR, MEM_SIZE, ADDRMODE_RELATIVE, -1, mem.socket);

  // tester
  dmaTester.outputSock.bind(*bus.t_sk[0]);
  dmaTester.mm2sIRQOnComp.bind(dmaCompleteSigs[0]);
  dmaTester.s2mmIRQOnComp.bind(dmaCompleteSigs[1]);

  // mm2s bindings
  bus.memmap(MM2S_BASE_ADDR, MM2S_REG_SIZE, ADDRMODE_RELATIVE, -1, mm2s.tgt_socket);
  mm2s.init_socket.bind(*bus.t_sk[1]);
  mm2s.irq.bind(dmaCompleteSigs[0]);

  sock2sig.inputSock.bind(mm2s.stream_socket);
  sock2sig.outputSig.bind(externalChannelWriteBus[0]);
  sock2sig.outputValid.bind(dmaDataReadyValidBus[0]);
  sock2sig.peripheralReady.bind(sam.channel_dma_periph_ready_valid[0]);
  externalChannelWriteBus[0] = 0xDEADBEEF;

  channelEnable.outputValid.bind(dmaDataReadyValidBus[0]);
  channelEnable.peripheralReady.bind(sam.channel_dma_periph_ready_valid[0]);

  // s2mm bindings
  bus.memmap(S2MM_BASE_ADDR, MM2S_REG_SIZE, ADDRMODE_RELATIVE, -1, s2mm.tgt_socket);
  s2mm.init_socket.bind(*bus.t_sk[2]);
  s2mm.irq.bind(dmaCompleteSigs[1]);

  sig2sock.outputSock.bind(s2mm.stream_socket);
  sig2sock.inputSig.bind(externalChannelReadBus[1]);
  sig2sock.inputReady.bind(dmaDataReadyValidBus[1]);
  sig2sock.peripheralValid.bind(sam.channel_dma_periph_ready_valid[1]);
  sig2sock.packetLength.bind(s2mm.packet_length);

  channelEnable.peripheralValid.bind(sam.channel_dma_periph_ready_valid[1]);
  channelEnable.inputReady.bind(dmaDataReadyValidBus[1]);

  for (size_t ii = 0; ii < sam.read_channel_data.size(); ii++) {
    sam.read_channel_data[ii][0].bind(externalChannelReadBus[ii]);
    sam.write_channel_data[ii][0].bind(externalChannelWriteBus[ii]);
  }

  sc_trace(tf, mm2s.irq, "dma-mm2s-ioc");
}

int DMA_TB::runTB() {
  bool success;
  success = validateWriteToSAM1D();

  if (success) {
    std::cout << "TEST SUCCEEDED";
  } else {
    std::cout << "TEST FAILED";
  }

  std::cout << std::endl;

  return 0;
}

bool DMA_TB::validateWriteToSAM1D() {
  std::cout << "Testing writing to SAM from memory..." << std::endl;
  control.set_reset(true);
  control.set_program(false);

  sc_start(1, sc_core::SC_NS);

  control.set_reset(false);

  vector<Descriptor_2D> tempProgram;
  Descriptor_2D generateWrite1DDescriptor1(1, 10, DescriptorState::GENERATE, 9, 1, 0, 0);
  Descriptor_2D suspendWriteDescriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);
  Descriptor_2D waitDescriptor(1, 10, DescriptorState::WAIT, 2, 5, 0, 0);
  Descriptor_2D generateRead1DDescriptor(2, 10, DescriptorState::GENERATE, 9, 1, 0, 0);
  Descriptor_2D suspendReadDescriptor(2, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);

  // Write program
  tempProgram.push_back(generateWrite1DDescriptor1);
  tempProgram.push_back(suspendWriteDescriptor);
  sam.generators[0].loadProgram(tempProgram);
  sam.channels[0].set_mode(MemoryChannelMode::WRITE);

  // Read program
  tempProgram.clear();
  tempProgram.push_back(waitDescriptor);
  tempProgram.push_back(generateRead1DDescriptor);
  tempProgram.push_back(suspendReadDescriptor);
  sam.generators[1].loadProgram(tempProgram);
  sam.channels[1].set_mode(MemoryChannelMode::READ);

  control.set_program(true);
  std::cout << "Load program and start first descriptor" << std::endl;
  sc_start(1, SC_NS);
  control.set_program(false);

  sc_start(50, SC_NS);

  size_t memIdx = 10;

  for (float ff : dmaTester.testData) {
    if (sam.mem.ram[memIdx][0] !=
        sc_int<32>(*reinterpret_cast<int32_t*>(&ff))) {  // This is gross but unfortunately we need
                                                         // to get the binary-equivalent integer
      std::cout << "sam.mem.ram[" << memIdx << "][0] != expected value: " << ff << std::endl;
      return false;
    }

    memIdx++;
  }
  std::cout << "Memory to stream operation validated" << std::endl;

  return true;
}

int sc_main(int argc, char* argv[]) {
  DMA_TB dmaTB;

  return dmaTB.runTB();
}