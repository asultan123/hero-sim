#include "bytePrinter.hh"

#include <iostream>

BytePrinter::BytePrinter(sc_module_name moduleName) : sc_module(moduleName) {
  SC_THREAD(printData);
}

void BytePrinter::printData() {
  while (true) {
    wait(dataReady.posedge_event());

    char value = inputSig.read();

    std::cout << value;

    assertRead = true;
    wait(1, SC_NS);
    assertRead = false;
  }
}