#include "dmaTester.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "dmaTestRegisterMap.hh"

DMATester::DMATester(bool burstMode, sc_module_name moduleName)
    : sc_module(moduleName), burstMode(burstMode), mm2sOffset(0), s2mmOffset(0) {
  SC_THREAD(testRun);
  SC_THREAD(verifyMemWriteback);
}

void DMATester::loadData() {
  for (size_t ii = 0; ii < 10; ii++) {
    testData.push_back(ii);
  }

  // Load to memory
  sc_time transportTime = SC_ZERO_TIME;
  trans.reset();
  trans.set_write();
  trans.set_address(MEM_BASE_ADDR);
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(testData.data()));
  trans.set_data_length(testData.size() * sizeof(float));
  outputSock->b_transport(trans, transportTime);
  if (!trans.is_response_ok()) throw std::runtime_error("Failed to write test data to memory");
  dataLoaded.notify();
}

void DMATester::sendDMAReq(bool s2mm) {
  // Set address to pull from
  sc_time transportTime = SC_ZERO_TIME;
  uint64_t dataAddress =
      MEM_BASE_ADDR + (s2mm ? (testData.size() + s2mmOffset) : mm2sOffset) * sizeof(float);
  uint32_t addrLsb = ((dataAddress << 32) >> 32);
  uint32_t addrMsb = (dataAddress >> 32);
  uint32_t dataSize = sizeof(float);
  if (burstMode) dataSize *= testData.size();
  trans.reset();
  trans.set_write();
  trans.set_address(s2mm ? S2MM_ADDR_REG(0) : MM2S_ADDR_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&addrLsb));
  trans.set_data_length(sizeof(addrLsb));
  trans.set_streaming_width(sizeof(addrLsb));
  outputSock->b_transport(trans, transportTime);
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  trans.set_address(s2mm ? S2MM_ADDR_MSB_REG(0) : MM2S_ADDR_MSB_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&addrMsb));
  trans.set_data_length(sizeof(addrMsb));
  trans.set_streaming_width(sizeof(addrMsb));
  outputSock->b_transport(trans, transportTime);

  // Write data length, start DMA
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  trans.set_address(s2mm ? S2MM_LENGTH_REG(0) : MM2S_LENGTH_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&dataSize));
  trans.set_data_length(sizeof(dataSize));
  outputSock->b_transport(trans, transportTime);
  if (s2mm)
    s2mmOffset += dataSize / sizeof(float);
  else
    mm2sOffset += dataSize / sizeof(float);
}

void DMATester::sendMM2SReq() { sendDMAReq(false); }

void DMATester::sendS2MMReq() { sendDMAReq(true); }

void DMATester::testRun() {
  wait(20, SC_NS);  // Wait for reset sequence to run
  loadData();

  enableIOC(false, 0);

  while (true) {
    if (mm2sOffset < testData.size())
      sendMM2SReq();
    else
      break;
    wait(mm2sIRQOnComp.posedge_event());

    clearIOC(false, 0);
  }

  std::cout << "Test data consumed!" << std::endl;
}

void DMATester::verifyMemWriteback() {
  wait(20, SC_NS);  // Wait for reset sequence to run

  enableIOC(true, 0);
  wait(dataLoaded);

  while (true) {
    if (s2mmOffset < testData.size())
      sendS2MMReq();
    else
      break;
    wait(s2mmIRQOnComp->posedge_event());

    clearIOC(true, 0);
  }

  // Read from memory
  std::vector<float> memData(testData.size());
  sc_time transportTime = SC_ZERO_TIME;
  trans.reset();
  trans.set_read();
  trans.set_address(MEM_BASE_ADDR + testData.size() * sizeof(float));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(memData.data()));
  trans.set_data_length(memData.size() * sizeof(float));
  outputSock->b_transport(trans, transportTime);
  if (!trans.is_response_ok())
    throw std::runtime_error("Failed to read writeback data from memory");

  for (size_t ii = 0; ii < testData.size(); ii++) {
    if (testData.at(ii) != memData.at(ii)) {
      std::cout << "Writeback data doesn't match test data" << std::endl;
      return;
    }
  }
}

void DMATester::enableIOC(bool s2mm, size_t ii) {
  sc_time transportTime = SC_ZERO_TIME;  // The b_transports don't use this anyways

  uint32_t crRegVal;
  trans.reset();
  trans.set_read();
  trans.set_address(s2mm ? S2MM_CR_REG(ii) : MM2S_CR_REG(ii));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&crRegVal));
  trans.set_data_length(sizeof(crRegVal));
  trans.set_streaming_width(sizeof(crRegVal));
  outputSock->b_transport(trans, transportTime);

  crRegVal |= AXIDMA_CR_IOC_IRQ_EN;

  trans.set_write();
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  outputSock->b_transport(trans, transportTime);
}

void DMATester::clearIOC(bool s2mm, size_t ii) {
  sc_time transportTime = SC_ZERO_TIME;  // The b_transports don't use this anyways
  // Get current status register value
  uint32_t srRegVal;
  trans.reset();
  trans.set_read();
  trans.set_address(s2mm ? S2MM_SR_REG(ii) : MM2S_SR_REG(ii));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&srRegVal));
  trans.set_data_length(sizeof(srRegVal));
  trans.set_streaming_width(sizeof(srRegVal));
  outputSock->b_transport(trans, transportTime);

  wait(50, SC_PS);  // So that we see something on the interrupt line

  // Clear complete interrupt (clears on write, just write the same thing
  // back)
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  trans.set_write();
  outputSock->b_transport(trans, transportTime);

  wait(50, SC_PS);  // Give time for IOC to be deasserted
}