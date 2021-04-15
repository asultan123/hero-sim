#include "bytePrinter.hh"

#include <iostream>

BytePrinter::BytePrinter(sc_clock& clk, sc_module_name moduleName)
    : sc_module(moduleName), clk(clk) {
  SC_METHOD(printData);
  sensitive << clk.posedge_event();
}

void BytePrinter::printData() {
  peripheralReady = true;
  if (outputValid->read()) {
    char value = inputSig.read();
    std::cout << value;
  }
}