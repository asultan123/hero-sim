#if !defined(__DMATESTER_H__)
#define __DMATESTER_H__

#include <tlm_utils/simple_initiator_socket.h>

#include <cstddef>
#include <systemc>
#include <vector>

using namespace sc_core;
using namespace sc_dt;

class DMATester : public sc_module {
  SC_HAS_PROCESS(DMATester);

 public:
  DMATester(bool burstMode = false, sc_module_name moduleName = "dma-tester");

  tlm_utils::simple_initiator_socket<DMATester> outputSock;
  sc_in<bool> mm2sIRQOnComp;  // Interrupt on complete
  sc_in<bool> s2mmIRQOnComp;
  std::vector<float> testData;

 private:
  void loadData();
  void sendDMAReq(bool s2mm);
  void sendMM2SReq();
  void sendS2MMReq();
  void testRun();
  void verifyMemWriteback();
  void enableIOC(bool s2mm, size_t ii);
  void clearIOC(bool s2mm, size_t ii);

  size_t mm2sOffset;
  size_t s2mmOffset;
  bool burstMode;
  tlm::tlm_generic_payload trans;
  sc_event dataLoaded;
};

#endif  // __DMATESTER_H__