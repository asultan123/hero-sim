#include <iostream>

#include "bytePrinter.hh"
#include "sock2sig.hh"
#include "stringProducer.hh"

int sc_main(int argc, char* argv[]) {
  Sock2Sig<8> adapter;
  BytePrinter printer;
  StringProducer producer;
  sc_signal<sc_int<8>> dataSig;
  sc_signal<bool> dataReadySig, assertReadSig;

  // Bindings
  printer.inputSig(dataSig);
  printer.assertRead(assertReadSig);
  printer.dataReady(dataReadySig);
  adapter.outputSig(dataSig);
  adapter.assertRead(assertReadSig);
  adapter.dataReady(dataReadySig);
  adapter.inputSock(producer.outputSock);

  std::cout << "START" << std::endl;

  sc_start(100, SC_NS);

  std::cout << std::endl;

  return 0;
}
