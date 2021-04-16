#if !defined(__AXIDMA_MEM_TB_H__)
#define __AXIDMA_MEM_TB_H__

#include <systemc>
#include <vector>

#include "SAM.hh"
#include "channelEnable.hh"
#include "dmaTester.hh"
#include "iconnect.h"
#include "memory.h"
#include "sig2sock.hh"
#include "sock2sig.hh"
#include "xilinx-axidma.h"

class DMA_TB : public sc_module {
  SC_HAS_PROCESS(DMA_TB);

 public:
  DMA_TB(sc_module_name moduleName = "dma-tb");

  sc_trace_file* tf;
  GlobalControlChannel control;
  SAM<sc_int<32>> sam;
  Sock2Sig<32> sock2sig;
  Sig2Sock<32> sig2sock;
  sc_vector<sc_signal<bool>> dmaCompleteSigs;
  axidma_mm2s mm2s;
  axidma_s2mm s2mm;
  iconnect<3, 3> bus;
  memory mem;
  DMATester dmaTester;
  sc_vector<sc_signal<sc_int<32>>> externalChannelReadBus;
  sc_vector<sc_signal<sc_int<32>, SC_MANY_WRITERS>> externalChannelWriteBus;
  sc_vector<sc_signal<bool>> dmaDataReadyValidBus;
  sc_vector<sc_signal<bool>> dmaPeriphReadyValidBus;
  ChannelEnable channelEnable;

  int runTB();

 private:
  bool validateWriteToSAM1D();
};

#endif  // __AXIDMA_MEM_TB_H__