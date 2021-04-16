#include "channelEnable.hh"

#include "GlobalControl.hh"

ChannelEnable::ChannelEnable(GlobalControlChannel_IF& control, sc_trace_file* tf,
                             sc_module_name moduleName)
    : sc_module(moduleName),
      enable("enable"),
      outputValid("output-valid"),
      peripheralReady("peripheral-ready"),
      peripheralValid("peripheral-valid"),
      inputReady("input-ready") {
  enable.bind(const_cast<sc_signal<bool, SC_MANY_WRITERS>&>(control.enable()));
  SC_METHOD(updateEnable);
  sensitive << outputValid.value_changed() << peripheralReady.value_changed()
            << peripheralValid.value_changed() << inputReady.value_changed();

  if (tf) {
    sc_trace(tf, enable, enable.name());
    sc_trace(tf, outputValid, outputValid.name());
    sc_trace(tf, peripheralReady, peripheralReady.name());
    sc_trace(tf, peripheralValid, peripheralValid.name());
    sc_trace(tf, inputReady, inputReady.name());
  }
}

void ChannelEnable::updateEnable() {
  enable = (outputValid && peripheralReady && !peripheralValid) ||
           (!peripheralReady && peripheralValid && inputReady) ||
           (outputValid && peripheralReady && peripheralValid && inputReady);
}