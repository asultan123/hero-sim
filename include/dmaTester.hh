/**
 * @file dmaTester.hh
 * @author Vincent Zhao
 * @brief Testbench for testing reading and writing data between memory and the SAM via the AXI DMA.
 */

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

  tlm_utils::simple_initiator_socket<DMATester>
      outputSock;               //! Initiates transactions to the memory bus
  sc_in<bool> mm2sIRQOnComp;    //! mm2s interrupt on complete
  sc_in<bool> s2mmIRQOnComp;    //! s2mm interrupt on complete
  std::vector<float> testData;  //! Test data for reading and writing verification

 private:
  void loadData();  //! Writes test data to memory for testing memory-to-stream reading
  void sendDMAReq(
      bool s2mm);             //! Starts the next transaction for the corresponding stream direction
  void sendMM2SReq();         //! Starts the next memory-to-stream transaction
  void sendS2MMReq();         //! Starts the next stream-to-memory transaction
  void testRun();             //! Performs memory-to-stream test
  void verifyMemWriteback();  //! Performs stream-to-memory test
  void enableIOC(bool s2mm, size_t ii);  //! Enables interrupt on complete on specified DMA
  void clearIOC(bool s2mm, size_t ii);   //! Clears raised interrupt on complete on specified DMA

  size_t mm2sOffset;  //! Where in the test data to start the next memory-to-stream transaction
  size_t s2mmOffset;  //! Where in the test data to start the next stream-to-memory transaction
  bool
      burstMode;  //! Whether to send the entire test data as one transaction or one float at a time
  tlm::tlm_generic_payload trans;  //! TLM tranaction buffer
  sc_event dataLoaded;             //! If the test data has been loaded into memory
};

#endif  // __DMATESTER_H__