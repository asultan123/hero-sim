#include <bits/stdint-uintn.h>
#include <sysc/utils/sc_vector.h>
#include <vector>
#if !defined(__AXIDMA_MEM_TB_H__)
#define __AXIDMA_MEM_TB_H__

#include <systemc>

#include "SAM.hh"
#include "dmaProducer.hh"
#include "iconnect.h"
#include "memory.h"
#include "sock2sig.hh"
#include "xilinx-axidma.h"

class DMA_TB : public sc_module {
public:
  DMA_TB(sc_module_name moduleName = "dma-tb");

  sc_trace_file *tf;
  GlobalControlChannel control;
  SAM<sc_int<32>> sam;
  Sock2Sig<32> sock2sig;
  sc_signal<bool> dmaCompleteSig;
  axidma_mm2s mm2s;
  iconnect<2, 2> bus;
  memory mem;
  DMAProducer producer;
  sc_vector<sc_signal<sc_int<32>>> externalChannelReadBus;
  sc_vector<sc_signal<sc_int<32>>> externalChannelWriteBus;
  sc_vector<sc_signal<bool>> dmaDataReadyBus;
  sc_vector<sc_signal<bool>> dmaAssertReadBus;

  int runTB();

private:
  bool validateWriteToSAM1D();
};

#endif // __AXIDMA_MEM_TB_H__