#if !defined(__DISSEMINATION_TB_H__)
#define __DISSEMINATION_TB_H__

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

template <typename T>
using SAMPtr = std::shared_ptr<SAM<T>>;

class DISSEMINATION_TB : public sc_module {
  SC_HAS_PROCESS(DISSEMINATION_TB);

 public:
  DISSEMINATION_TB(sc_module_name moduleName = "dissemination-tb");

  sc_trace_file* tf;
  GlobalControlChannel control;
  std::vector<SAMPtr<sc_int<32>>> sams;
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
  sc_vector<sc_signal<bool>> dmaTLastBus;
  ChannelEnable channelEnable;

  int runTB();

 private:
  bool validateWriteToSAM1D();
};

#endif  // __DISSEMINATION_TB_H__