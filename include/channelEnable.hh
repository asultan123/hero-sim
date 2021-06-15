/**
 * @file channelEnable.hh
 * @author Vincent Zhao
 * @brief Determines whether to enable an address generator based on the output valid, input ready,
 * and T last signals.
 */

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

  sc_out<bool> enable;      //! Enables the address generator
  sc_in<bool> outputValid;  //! Whether the mm2s DMA is providing valid data or is stale
  sc_in<bool> inputReady;   //! Whether the s2mm DMA is able to receive new data
  sc_in<bool>
      tLast;  //! If the last packet in the mm2s DMA for the current transaction has been reached

 private:
  void updateEnable();  //! Updates enable signal state
};

#endif  // __CHANNELENABLE_H__