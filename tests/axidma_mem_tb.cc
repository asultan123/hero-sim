#include "axidma_mem_tb.hh"

#include <sysc/communication/sc_signal.h>
#include <sysc/communication/sc_writer_policy.h>
#include <sysc/kernel/sc_time.h>

#include "dmaTestRegisterMap.hh"

namespace {
constexpr unsigned int dutMemLength = 128;
constexpr unsigned int dutMemWidth = 1;
constexpr unsigned int dutMemChannelCount = 1;
}  // namespace

DMA_TB::DMA_TB(sc_module_name moduleName)
    : sc_module(moduleName),
      tf(sc_create_vcd_trace_file("ProgTrace")),
      control("global-control-channel", sc_time(1, SC_NS), tf),
      sam("sam", control, dutMemChannelCount, dutMemLength, dutMemWidth, tf),
      sock2sig(control.clk()),
      mm2s("axidma-mm2s"),
      bus("bus"),
      mem("memory", SC_ZERO_TIME, MEM_SIZE),
      externalChannelReadBus("ext-channel-read-bus", dutMemWidth),
      externalChannelWriteBus("ext-channel-write-bus", dutMemWidth),
      dmaOutputValidBus("dma-data-ready-bus", dutMemWidth),
      dmaPeriphReadyBus("dma-assert-read-bus", dutMemWidth) {
  // Bindings

  bus.memmap(MEM_BASE_ADDR, MEM_SIZE, ADDRMODE_RELATIVE, -1, mem.socket);

  // producer
  producer.outputSock(*bus.t_sk[0]);

  producer.dmaIRQOnComp(dmaCompleteSig);
  mm2s.irq(dmaCompleteSig);

  // mm2s bindings
  bus.memmap(MM2S_BASE_ADDR, MM2S_REG_SIZE, ADDRMODE_RELATIVE, -1,
             mm2s.tgt_socket);
  mm2s.init_socket(*bus.t_sk[1]);

  sock2sig.inputSock(mm2s.stream_socket);
  sock2sig.outputSig(externalChannelWriteBus[0]);
  sock2sig.outputValid.bind(
      const_cast<sc_signal<bool, SC_MANY_WRITERS>&>(sam.control->enable()));
  sock2sig.peripheralReady(sam.write_channel_dma_periph_ready[0]);

  for (size_t ii = 0; ii < sam.read_channel_data[0].size(); ii++) {
    sam.read_channel_data[0][ii](externalChannelReadBus[ii]);
    sam.write_channel_data[0][ii](externalChannelWriteBus[ii]);
  }
}

int DMA_TB::runTB() {
  bool success;
  success = validateWriteToSAM1D();

  return 0;
}

bool DMA_TB::validateWriteToSAM1D() {
  std::cout << "Testing writing to SAM from memory..." << std::endl;
  control.set_reset(true);
  control.set_program(false);
  control.set_enable(false);

  sc_start(1, sc_core::SC_NS);

  control.set_reset(false);

  Descriptor_2D generate1DDescriptor1(1, 10, DescriptorState::GENERATE, 10, 1,
                                      0, 0);
  Descriptor_2D suspendDescriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);

  vector<Descriptor_2D> tempProgram{generate1DDescriptor1, suspendDescriptor};

  sam.generators[0].loadProgram(tempProgram);
  sam.channels[0].set_mode(MemoryChannelMode::WRITE);

  control.set_program(true);
  std::cout << "Load program and start first descriptor" << std::endl;
  sc_start(1, SC_NS);
  control.set_enable(true);
  control.set_program(false);

  sc_start(1000, SC_US);

  return true;
}

int sc_main(int argc, char* argv[]) {
  DMA_TB dmaTB;

  return dmaTB.runTB();
}