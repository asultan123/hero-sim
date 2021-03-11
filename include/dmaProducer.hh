#include <cstddef>
#if !defined(__DMAPRODUCER_H__)
#define __DMAPRODUCER_H__

#include <tlm_utils/simple_initiator_socket.h>

#include <systemc>

using namespace sc_core;
using namespace sc_dt;

class DMAProducer : public sc_module {
  SC_HAS_PROCESS(DMAProducer);

public:
  DMAProducer(sc_module_name moduleName = "dma-producer");

  tlm_utils::simple_initiator_socket<DMAProducer> outputSock;
  sc_in<bool> dmaIRQOnComp; // Interrupt on complete

private:
  void loadData();
  void sendDMAReq();
  void testRun();
  std::vector<float> testData;
  size_t currentTestData;

  tlm::tlm_generic_payload trans;
};

#endif // __DMAPRODUCER_H__