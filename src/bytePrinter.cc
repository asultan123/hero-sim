/**
 * @file bytePrinter.cc
 * @author Vincent Zhao
 * @brief Implementation of a simple module that prints out data from an input signal if the data is
 * valid.
 */

#include "bytePrinter.hh"

#include <iostream>

/**
 * @brief Construct a new Byte Printer:: Byte Printer object
 *
 * @param clk The clock signal this module is sensitive to.
 * @param moduleName The name of this module for identification, defaults to "byte-printer".
 */
BytePrinter::BytePrinter(sc_clock& clk, sc_module_name moduleName)
    : sc_module(moduleName), clk(clk) {
  SC_METHOD(printData);
  sensitive << clk.posedge_event();
}

/**
 * @brief Prints data received as chars if the data is marked as valid by the supplier.
 */
void BytePrinter::printData() {
  peripheralReady = true;
  if (outputValid->read()) {
    char value = inputSig.read();
    std::cout << value;
  }
}