#if !defined(__CHANNELENABLE_H__)
#define __CHANNELENABLE_H__

#include <systemc>

using namespace sc_core;

// Forward declarations
struct GlobalControlChannel_IF;

class ChannelEnable : public sc_module {
  SC_HAS_PROCESS(ChannelEnable);

 public:
  ChannelEnable(GlobalControlChannel_IF& control, sc_trace_file* tf = nullptr,
                sc_module_name moduleName = "channel-enable");

  sc_out<bool> enable;
  sc_in<bool> outputValid;
  sc_in<bool> peripheralReady;
  sc_in<bool> peripheralValid;
  sc_in<bool> inputReady;

 private:
  void updateEnable();
};

#endif  // __CHANNELENABLE_H__