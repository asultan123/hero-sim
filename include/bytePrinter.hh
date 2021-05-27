/**
 * @file bytePrinter.hh
 * @author Vincent Zhao
 * @brief A simple module that prints out data from an input signal if the data is valid.
 */

#if !defined(__BYTEPRINTER_H__)
#define __BYTEPRINTER_H__

#include <systemc>

using namespace sc_core;
using namespace sc_dt;

class BytePrinter : public sc_module {
  SC_HAS_PROCESS(BytePrinter);

 public:
  BytePrinter(sc_clock& clk, sc_module_name moduleName = "byte-printer");

  sc_in<sc_int<8>> inputSig;  //! Input data to receive
  sc_in<bool> outputValid;    //! Whether the data supplied is still valid or is stale
  sc_out<bool>
      peripheralReady;  //! Lets data supplier know that module is ready to receive new data

 private:
  void printData();  //! Prints received data as chars

  sc_in<bool> clk;  //! Clock, data is printed on positive edges
};

#endif  // __BYTEPRINTER_H__