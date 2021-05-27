/**
 * @file channelEnable.cc
 * @author Vincent Zhao
 * @brief Implementation of address generator enable control logic.
 */

#include "channelEnable.hh"

#include "GlobalControl.hh"

/**
 * @brief Construct a new ChannelEnable::ChannelEnable object
 *
 * @param control The global enable channel for the address generator, supplies the output enable
 * signal.
 * @param tf Traces signal state changes.
 * @param moduleName Name of this module instance for identification, defaults to "channel-enable".
 */
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

/**
 * @brief Changes the enable state based on the output valid, T last, and input ready signals on any
 * changes to the inputs.
 */
void ChannelEnable::updateEnable() { enable = (outputValid || tLast) && inputReady; }