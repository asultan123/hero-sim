#if !defined(__SOCK2SIG_H__)
#define __SOCK2SIG_H__

/**
 * @file sock2sig.hh
 * @author Vincent Zhao
 * @brief Converts data received from a TLM-2.0 socket to individual bit
 * signals, adds a data ready-read interface with adjustable timing.
 *
 *
 */

#include <tlm_utils/simple_target_socket.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <systemc>
#include <vector>

// Forward declarations
struct GlobalControlChannel_IF;

using namespace sc_core;
using namespace sc_dt;

typedef struct DataTrans {
  std::unique_ptr<std::vector<uint8_t>> data;
  bool last;  // The last transaction of a DMA operation
} DataTrans;

typedef std::unique_ptr<DataTrans> DataTransPtr;

template <unsigned int BUSWIDTH>
class Sock2Sig : public sc_module {
  SC_HAS_PROCESS(Sock2Sig<BUSWIDTH>);

 public:
  Sock2Sig(sc_clock& clk, size_t maxWords = 256, sc_trace_file* tf = nullptr,
           sc_module_name moduleName = "sock-2-sig");
  Sock2Sig(GlobalControlChannel_IF& control, size_t maxWords = 256, sc_trace_file* tf = nullptr,
           sc_module_name moduleName = "sock-2-sig");

  tlm_utils::simple_target_socket<Sock2Sig, BUSWIDTH> inputSock;
  sc_out<sc_int<BUSWIDTH>> outputSig;
  sc_out<bool> outputValid;     // Data is fresh, has not already been read
  sc_in<bool> peripheralReady;  // High if the peripheral is reading data every cycle
  sc_out<bool> tLast;           // The last transaction of a DMA operation was reached

 private:
  void inputSock_b_transport(tlm::tlm_generic_payload& trans, sc_time& delay);
  void updateOutput();

  std::queue<DataTransPtr> buffer;
  DataTransPtr currentData;
  size_t bitOffset;
  size_t byteOffset;
  size_t currentWords;
  size_t maxWords;
  bool isInit;
  sc_event wordRead;
  sc_in<bool> clk;
};

#endif  // __SOCK2SIG_H__
