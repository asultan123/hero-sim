#include "channelEnable.hh"

#include <sysc/tracing/sc_trace.h>

#include "GlobalControl.hh"

ChannelEnable::ChannelEnable(GlobalControlChannel_IF& control, sc_trace_file* tf,
                             sc_module_name moduleName)
    : sc_module(moduleName),
      enable("enable"),
      outputValid("output-valid"),
      inputReady("input-ready"),
      tLast("t-last") {
  enable.bind(const_cast<sc_signal<bool, SC_MANY_WRITERS>&>(control.enable()));
  SC_METHOD(updateEnable);
  sensitive << outputValid.value_changed() << inputReady.value_changed() << tLast.value_changed();

  if (tf) {
    sc_trace(tf, enable, enable.name());
    sc_trace(tf, outputValid, outputValid.name());
    sc_trace(tf, inputReady, inputReady.name());
    sc_trace(tf, tLast, tLast.name());
  }
}

void ChannelEnable::updateEnable() { enable = (outputValid || tLast) && inputReady; }