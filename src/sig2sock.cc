/**
 * @file sig2sock.cc
 * @author Vincent Zhao
 * @brief Converts data received from individual bit
 * signals to a TLM-2.0 socket, adds a data ready-read interface with adjustable timing.
 */

#include "sig2sock.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "GlobalControl.hh"

namespace {
inline uint64_t bitsToBytes(unsigned int bits) {
  uint64_t bytes = bits / 8;
  return bits % 8 ? bytes + 1 : bytes;
}
}  // namespace

/**
 * @brief Construct a new Sig2Sock<BUSWIDTH>::Sig2Sock object
 *
 * @tparam BUSWIDTH How many bits wide a word is
 * @param clk Clock signal
 * @param maxWords Maximum number of words that can be stored in the FIFO buffer
 * @param moduleName Module identifier
 * @param tf Signal state trace file
 */
template <unsigned int BUSWIDTH>
Sig2Sock<BUSWIDTH>::Sig2Sock(sc_clock& clk, size_t maxWords, sc_module_name moduleName,
                             sc_trace_file* tf)
    : sc_module(moduleName),
      inputSig("input-sig"),
      peripheralValid("peripherhal-valid"),
      packetLength("packet-length"),
      currentWords(0),
      maxWords(maxWords),
      clk(clk),
      setupTime(1) {
  if (BUSWIDTH % 8 != 0)
    throw std::runtime_error("Adapter does not currently support non-byte aligned widths");
  SC_METHOD(updateInput);
  sensitive << clk.posedge_event();
  SC_METHOD(resetSetupTime);
  sensitive << peripheralValid.neg();

  if (tf) {
    sc_trace(tf, inputSig, inputSig.name());
    sc_trace(tf, peripheralValid, peripheralValid.name());
    sc_trace(tf, packetLength, packetLength.name());
  }
}

/**
 * @brief Construct a new Sig2Sock<BUSWIDTH>::Sig2Sock object using a global control channel
 *
 * @tparam BUSWIDTH How many bits wide a word is
 * @param control Global control channel, provides clock signal
 * @param maxWords Maximum number of words that can be stored in the FIFO buffer
 * @param tf Signal state trace file
 * @param moduleName Module identifier
 */
template <unsigned int BUSWIDTH>
Sig2Sock<BUSWIDTH>::Sig2Sock(GlobalControlChannel_IF& control, size_t maxWords, sc_trace_file* tf,
                             sc_module_name moduleName)
    : Sig2Sock<BUSWIDTH>(control.clk(), maxWords, moduleName, tf) {}

/**
 * @brief Reads input data into FIFO buffer when needed.
 *
 * Setup time must be fulfilled, space must be available in the buffer, and the peripheral must be
 * valid.
 *
 * @tparam BUSWIDTH How many bits wide a word is
 */
template <unsigned int BUSWIDTH>
void Sig2Sock<BUSWIDTH>::updateInput() {
  if (peripheralValid->read() && currentWords < maxWords) {
    if (setupTime) {
      setupTime--;
      return;
    }

    buffer.push(inputSig.read());
    currentWords++;
  }

  flushBuffer();
  inputReady = currentWords < maxWords;
}

/**
 * @brief Writes data in FIFO to destination memory.
 *
 * Buffered data size must be equal to or greater than the length supplied by the stream-to-memory
 * DMA.
 *
 * @tparam BUSWIDTH How many bits wide a word is
 */
template <unsigned int BUSWIDTH>
void Sig2Sock<BUSWIDTH>::flushBuffer() {
  size_t requiredLength = packetLength.read();
  if (requiredLength && buffer.size() * bitsToBytes(BUSWIDTH) >= requiredLength) {
    std::vector<uint8_t> transData;
    sc_time transportTime = SC_ZERO_TIME;

    for (size_t ii = 0; ii < requiredLength; ii++) {
      assert(bitsToBytes(BUSWIDTH) <= sizeof(uint64_t));
      uint8_t* value = reinterpret_cast<uint8_t*>(&buffer.front());
      for (uint8_t ii = 0; ii < bitsToBytes(BUSWIDTH); ii++) {
        transData.push_back(*value);
        value++;
      }
      buffer.pop();
    }
    trans.reset();
    trans.set_write();
    trans.set_data_ptr(transData.data());
    trans.set_data_length(transData.size());
    // NOTE: eop is set by default, not needed to set explicitly in extension because sig2sock only
    // sends full packets
    outputSock->b_transport(trans, transportTime);
    if (!trans.is_response_ok())
      throw std::runtime_error("Failed to write DMA transaction to memory");
  }
}

/**
 * @brief Resets setup time counter
 *
 * @tparam BUSWIDTH How many bits wide a word is
 */
template <unsigned int BUSWIDTH>
void Sig2Sock<BUSWIDTH>::resetSetupTime() {
  setupTime = 1;
}

template class Sig2Sock<8>;
template class Sig2Sock<32>;
template class Sig2Sock<64>;