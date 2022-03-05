#if !defined(__BYTEPRINTER_H__)
#define __BYTEPRINTER_H__

#include <systemc>

using namespace sc_core;
using namespace sc_dt;

class BytePrinter : public sc_module
{
    SC_HAS_PROCESS(BytePrinter);

  public:
    BytePrinter(sc_module_name moduleName = "byte-printer");

    sc_in<sc_int<8>> inputSig;
    sc_in<bool> dataReady;
    sc_out<bool> assertRead;

  private:
    void printData();
};

#endif // __BYTEPRINTER_H__