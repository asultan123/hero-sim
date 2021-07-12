/**
 * @file dissemination_tb.hh
 * @author Vincent Zhao
 * @brief Basic test of program dissemination on a simple memory layout.
 *
 * Sends a set of distinct programs to a set of connected memories with address generators. The
 * layout of the memory is one parent SAM with 3 children SAMs attached.
 */

#if !defined(__DISSEMINATION_TB_H__)
#define __DISSEMINATION_TB_H__

#include <systemc>
#include <vector>

#include "MultiProgramSender.hh"
#include "SAM.hh"
#include "channelEnable.hh"
#include "dmaTester.hh"
#include "iconnect.h"
#include "memory.h"
#include "sig2sock.hh"
#include "sock2sig.hh"
#include "xilinx-axidma.h"

class DISSEMINATION_TB : public sc_module {
  SC_HAS_PROCESS(DISSEMINATION_TB);

 public:
  DISSEMINATION_TB(sc_module_name moduleName = "dissemination-tb");

  sc_trace_file* tf;
  GlobalControlChannel control;
  sc_vector<SAM<sc_int<32>>> sams;
  sc_vector<sc_signal<bool>> dmaCompleteSigs;
  sc_vector<sc_signal<sc_int<32>>> externalChannelReadBus;
  sc_vector<sc_signal<sc_int<32>, SC_MANY_WRITERS>> externalChannelWriteBus;
  sc_vector<sc_signal<bool>> dmaDataReadyValidBus;
  sc_vector<sc_signal<bool>> dmaPeriphReadyValidBus;
  sc_vector<sc_signal<bool>> dmaTLastBus;
  sc_vector<sc_signal<sc_int<32>>>
      programSigs;  //! Interconnects memories to transmit program bitstream
  sc_vector<sc_signal<sc_int<32>>>
      dummySigs;  //! Dummy signals to fulfill module connection requirements
  ChannelEnable channelEnable;
  MultiProgramSender<sc_int<32>> programSender;  //! Packages and sends out programs as a bitstream

  int runTB();

 private:
  bool disseminatePrograms();  //! Sends programs to each address generator
};

#endif  // __DISSEMINATION_TB_H__