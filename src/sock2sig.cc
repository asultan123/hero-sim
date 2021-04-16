#include "sock2sig.hh"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

#include "GlobalControl.hh"

namespace {
inline uint64_t bitsToBytes(unsigned int bits) {
  uint64_t bytes = bits / 8;
  return bits % 8 ? bytes + 1 : bytes;
}
}  // namespace

template <unsigned int BUSWIDTH>
Sock2Sig<BUSWIDTH>::Sock2Sig(sc_clock& clk, size_t maxWords, sc_trace_file* tf,
                             sc_module_name moduleName)
    : sc_module(moduleName),
      outputSig("output-sig"),
      outputValid("output-valid"),
      peripheralReady("peripherhal-ready"),
      bitOffset(0),
      byteOffset(0),
      currentWords(0),
      maxWords(maxWords),
      isInit(false),
      clk(clk),
      setupTime(1) {
  if (BUSWIDTH % 8 != 0)
    throw std::runtime_error("Adapter does not currently support non-byte aligned widths");
  inputSock.register_b_transport(this, &Sock2Sig<BUSWIDTH>::inputSock_b_transport);
  SC_METHOD(updateOutput);
  sensitive << clk.posedge_event();

  if (tf) {
    sc_trace(tf, outputSig, outputSig.name());
    sc_trace(tf, outputValid, outputValid.name());
    sc_trace(tf, peripheralReady, peripheralReady.name());
  }
}

template <unsigned int BUSWIDTH>
Sock2Sig<BUSWIDTH>::Sock2Sig(GlobalControlChannel_IF& control, size_t maxWords, sc_trace_file* tf,
                             sc_module_name moduleName)
    : Sock2Sig<BUSWIDTH>(control.clk(), maxWords, tf, moduleName) {}

template <unsigned int BUSWIDTH>
void Sock2Sig<BUSWIDTH>::inputSock_b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
  // Should only propagate writes
  if (trans.get_command() != tlm::tlm_command::TLM_WRITE_COMMAND || trans.get_data_length() <= 0) {
    std::cout << "Transaction is invalid, discarding..." << std::endl;
    return;
  }

  size_t wordsReceived = trans.get_data_length() / bitsToBytes(BUSWIDTH);
  if (wordsReceived % bitsToBytes(BUSWIDTH) != 0) wordsReceived++;

  if (wordsReceived > maxWords) {
    std::cout << "Transaction is too large" << std::endl;
    trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
    return;
  }

  // Cache received data for later (may be wider than the bus)
  auto receivedData = std::make_unique<std::vector<uint8_t>>(trans.get_data_length());
  memcpy(receivedData->data(), trans.get_data_ptr(), receivedData->size());

  // Wait if no space in buffer
  while (currentWords + wordsReceived > maxWords) wait(wordRead);

  buffer.push(std::move(receivedData));

  currentWords += wordsReceived;

  trans.set_response_status(tlm::TLM_OK_RESPONSE);
}

template <unsigned int BUSWIDTH>
void Sock2Sig<BUSWIDTH>::updateOutput() {
  if (!currentData) {
    if (buffer.empty()) {
      if (holdTime) {
        holdTime--;
        return;
      }
      outputValid = false;
      return;
    }

    currentData = std::move(buffer.front());
    buffer.pop();

    byteOffset = 0;
    bitOffset = 0;
  }

  // We only bother updating the output if a peripheral is available to read it
  if (isInit) {
    if (setupTime) {
      setupTime--;
      return;
    }
    if (!peripheralReady->read()) return;
  } else {
    isInit = true;
    holdTime = 2;
  }

  // Copy the smaller of either the closest number of bytes that fits bus
  // width or bytes left in packet
  size_t bytesToCopy = std::min(bitsToBytes(BUSWIDTH), currentData->size() - byteOffset);

  uint64_t value = 0;

  memcpy(&value, &(*currentData)[byteOffset], bytesToCopy);

  // TODO: Non-byte aligned widths, revisit when needed
  // // Trim bits already read
  // value <<= bitOffset;

  // // Trim trailing bits
  // value >>= (64 - BUSWIDTH);
  // value <<= (64 - BUSWIDTH);
  // bitOffset = (bitOffset + BUSWIDTH) % 8;

  outputSig = sc_int<BUSWIDTH>(value);
  outputValid = true;
  wordRead.notify();
  currentWords--;

  byteOffset += (bitOffset + BUSWIDTH) / 8;

  // currentData consumed, discard
  if (byteOffset >= currentData->size()) currentData.reset();
}

template class Sock2Sig<8>;
template class Sock2Sig<32>;
template class Sock2Sig<64>;