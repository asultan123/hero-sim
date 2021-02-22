/**
 * @file sock2sig.hh
 * @author Vincent Zhao
 * @brief Converts data received from a TLM-2.0 socket to individual bit
 * signals, adds a data ready-read interface with adjustable timing.
 *
 *
 */

#if !defined(__SOCK2SIG_H__)
#define __SOCK2SIG_H__

#include <tlm_utils/simple_target_socket.h>

#include <memory>
#include <queue>
#include <systemc>

using namespace sc_core;
using namespace sc_dt;

template <unsigned int BUSWIDTH>
class Sock2Sig : public sc_module {
  SC_HAS_PROCESS(Sock2Sig<BUSWIDTH>);

 public:
  Sock2Sig(int readyDelay = 1, sc_module_name moduleName = "sock-2-sig");

  tlm_utils::simple_target_socket<Sock2Sig, BUSWIDTH> inputSock;
  sc_out<sc_int<BUSWIDTH>> outputSig;
  sc_out<bool> dataReady;
  sc_in<bool> assertRead;

 private:
  void inputSock_b_transport(tlm::tlm_generic_payload& trans, sc_time& delay);
  void updateOutput();

  std::queue<std::unique_ptr<std::vector<uint8_t>>> buffer;
  std::unique_ptr<std::vector<uint8_t>> currentData;
  ssize_t bitOffset;
  ssize_t byteOffset;
  sc_event dataAvailable;
  int readyDelay;
};

#endif  // __SOCK2SIG_H__
