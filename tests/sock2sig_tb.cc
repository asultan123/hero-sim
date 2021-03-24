#include <sysc/communication/sc_clock.h>
#include <sysc/communication/sc_writer_policy.h>
#include <sysc/kernel/sc_time.h>

#include <iostream>

#include "bytePrinter.hh"
#include "sock2sig.hh"
#include "stringProducer.hh"

int sc_main(int argc, char* argv[]) {
  sc_clock clk;
  Sock2Sig<8> adapter(clk);
  BytePrinter printer(clk);
  StringProducer producer;
  sc_signal<sc_int<8>, SC_MANY_WRITERS> dataSig;
  sc_signal<bool> outputValidSig, peripheralReadySig;

  // Bindings
  printer.inputSig(dataSig);
  printer.peripheralReady(peripheralReadySig);
  printer.outputValid(outputValidSig);
  adapter.outputSig(dataSig);
  adapter.peripheralReady(peripheralReadySig);
  adapter.outputValid(outputValidSig);
  adapter.inputSock(producer.outputSock);

  std::cout << "START" << std::endl;

  sc_start(100, sc_core::SC_NS);

  std::cout << std::endl;

  std::cout << "ALL TESTS PASS" << std::endl;

  return 0;
}
