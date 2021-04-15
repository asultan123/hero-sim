#include "dmaProducer.hh"

#include <bits/stdint-uintn.h>
#include <sys/types.h>
#include <sysc/kernel/sc_time.h>
#include <tlm_core/tlm_2/tlm_generic_payload/tlm_gp.h>

#include <cstddef>
#include <stdexcept>

#include "dmaTestRegisterMap.hh"

DMAProducer::DMAProducer(sc_module_name moduleName, bool burstMode)
    : sc_module(moduleName), burstMode(burstMode) {
  SC_THREAD(testRun);
}

void DMAProducer::loadData() {
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
  if (!trans.is_response_ok())
    throw std::runtime_error("Failed to write test data to memory");
}

void DMAProducer::sendDMAReq() {
  // Set address to pull from
  sc_time transportTime = SC_ZERO_TIME;
  uint64_t dataAddress = MEM_BASE_ADDR + currentTestData * sizeof(float);
  uint32_t addrLsb = ((dataAddress << 32) >> 32);
  uint32_t addrMsb = (dataAddress >> 32);
  uint32_t dataSize = sizeof(float);
  if (burstMode) dataSize *= testData.size();
  trans.reset();
  trans.set_write();
  trans.set_address(MM2S_ADDR_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&addrLsb));
  trans.set_data_length(sizeof(addrLsb));
  trans.set_streaming_width(sizeof(addrLsb));
  outputSock->b_transport(trans, transportTime);
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  trans.set_address(MM2S_ADDR_MSB_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&addrMsb));
  trans.set_data_length(sizeof(addrMsb));
  trans.set_streaming_width(sizeof(addrMsb));
  outputSock->b_transport(trans, transportTime);

  // Write data length, start DMA (one float at a time for now)
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  trans.set_address(MM2S_LENGTH_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&dataSize));
  trans.set_data_length(sizeof(dataSize));
  outputSock->b_transport(trans, transportTime);
  currentTestData += dataSize / sizeof(float);
}

void DMAProducer::testRun() {
  wait(20, SC_NS);  // Wait for reset sequence to run
  loadData();

  sc_time transportTime =
      SC_ZERO_TIME;  // The b_transports don't use this anyways

  // Enable Interrupt on Complete on DMA
  uint32_t crRegVal;
  trans.reset();
  trans.set_read();
  trans.set_address(MM2S_CR_REG(0));
  trans.set_data_ptr(reinterpret_cast<uint8_t*>(&crRegVal));
  trans.set_data_length(sizeof(crRegVal));
  trans.set_streaming_width(sizeof(crRegVal));
  outputSock->b_transport(trans, transportTime);

  crRegVal |= AXIDMA_CR_IOC_IRQ_EN;

  trans.set_write();
  trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
  outputSock->b_transport(trans, transportTime);

  while (true) {
    if (currentTestData < testData.size())
      sendDMAReq();
    else
      break;
    wait(dmaIRQOnComp.posedge_event());

    // Get current status register value
    uint32_t srRegVal;
    trans.reset();
    trans.set_read();
    trans.set_address(MM2S_SR_REG(0));
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

  std::cout << "Test data consumed!" << std::endl;
}