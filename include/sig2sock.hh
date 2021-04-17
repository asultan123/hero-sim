#if !defined(__SIG2SOCK_H__)
#define __SIG2SOCK_H__

/**
 * @file sig2sock.hh
 * @author Vincent Zhao
 * @brief Converts data received from individual bit
 * signals to a TLM-2.0 socket, adds a data ready-read interface with adjustable timing.
 */

#include <tlm_utils/simple_initiator_socket.h>

#include <cstddef>
#include <cstdint>
#include <queue>
#include <systemc>

using namespace sc_core;
using namespace sc_dt;

// Forward declarations
struct GlobalControlChannel_IF;

template <unsigned int BUSWIDTH>
class Sig2Sock : public sc_module {
  SC_HAS_PROCESS(Sig2Sock<BUSWIDTH>);

 public:
  Sig2Sock(sc_clock& clk, size_t maxWords = 256, sc_module_name moduleName = "sig-2-sock",
           sc_trace_file* tf = nullptr);
  Sig2Sock(GlobalControlChannel_IF& control, size_t maxWords = 256, sc_trace_file* tf = nullptr,
           sc_module_name moduleName = "sig-2-sock");

  tlm_utils::simple_initiator_socket<Sig2Sock, BUSWIDTH> outputSock;
  sc_in<sc_int<BUSWIDTH>> inputSig;
  sc_out<bool> inputReady;      // High if buffer has space for new data
  sc_in<bool> peripheralValid;  // High if the peripheral is supplying fresh data for the current
                                // clock cycle
  sc_in<uint32_t> packetLength;

 private:
  void updateInput();
  void flushBuffer();
  void resetSetupTime();

  std::queue<uint64_t> buffer;
  size_t currentWords;
  size_t maxWords;
  sc_in<bool> clk;
  tlm::tlm_generic_payload trans;
  size_t setupTime;
};

#endif  // __SIG2SOCK_H__
