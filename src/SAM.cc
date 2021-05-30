#include "SAM.hh"

template <typename DataType>
SAMDataPortCreator<DataType>::SAMDataPortCreator(unsigned int _width, sc_trace_file* _tf)
    : tf(_tf), width(_width) {}

template <typename DataType>
sc_vector<DataType>* SAMDataPortCreator<DataType>::operator()(const char* name, size_t) {
  return new sc_vector<DataType>(name, width);
}

template <typename DataType>
void SAM<DataType>::update() {
  if (control->reset()) {
  } else if (control->enable()) {
    in_port_propogate();
  }
}

template <typename DataType>
void SAM<DataType>::in_port_propogate() {
  if (control->enable()) {
    for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++) {
      for (unsigned int bus_index = 0; bus_index < width; bus_index++) {
        channels[channel_index].channel_write_data_element(
            write_channel_data[channel_index][bus_index].read(), bus_index);
      }
    }
  }
}

template <typename DataType>
void SAM<DataType>::out_port_propogate() {
  if (control->enable()) {
    for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++) {
      for (unsigned int bus_index = 0; bus_index < width; bus_index++) {
        read_channel_data[channel_index][bus_index] =
            channels[channel_index].get_channel_read_data_bus()[bus_index];
      }
    }
  }
}

// Constructor
template <typename DataType>
SAM<DataType>::SAM(sc_module_name name, GlobalControlChannel& _control, unsigned int _channel_count,
                   unsigned int _length, unsigned int _width, sc_trace_file* tf,
                   uint16_t _start_uid, uint16_t _end_uid)
    : sc_module(name),
      mem("mem", _control, _channel_count, _length, _width, tf),
      generators("generator", _channel_count,
                 AddressGeneratorCreator<DataType>(_control, tf, _start_uid, _end_uid)),
      channels("channels", _channel_count, MemoryChannelCreator<DataType>(_width, tf)),
      channel_dma_periph_ready_valid("channel_dma_periph_ready_valid", _channel_count),
      read_channel_data("read_channel_data", _channel_count,
                        OutDataPortCreator<DataType>(_width, tf)),
      write_channel_data("write_channel_data", _channel_count,
                         InDataPortCreator<DataType>(_width, tf)),
      program_in("program_in"),
      program_out("program_out"),
      length(_length),
      width(_width),
      channel_count(_channel_count) {
  control(_control);
  _clk(control->clk());

  SC_METHOD(update);
  sensitive << _clk.pos();
  sensitive << control->reset();

  SC_METHOD(in_port_propogate);

  for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++) {
    for (unsigned int data_index = 0; data_index < width; data_index++) {
      sensitive << write_channel_data[channel_index][data_index];
      sc_trace(tf, write_channel_data[channel_index][data_index],
               write_channel_data[channel_index][data_index].name());
    }
  }

  SC_METHOD(out_port_propogate)

  for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++) {
    for (unsigned int data_index = 0; data_index < width; data_index++) {
      sensitive << channels[channel_index].get_channel_read_data_bus()[data_index];
      sc_trace(tf, read_channel_data[channel_index][data_index],
               read_channel_data[channel_index][data_index].name());
    }
  }

  for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++) {
    generators[channel_index].channel(channels.at(channel_index));
    mem.channels[channel_index](channels.at(channel_index));
    generators[channel_index].ready(channel_dma_periph_ready_valid.at(channel_index));
    sc_trace(tf, channel_dma_periph_ready_valid[channel_index],
             channel_dma_periph_ready_valid[channel_index].name());
  }

  cout << " SAM MODULE: " << name << " has been instantiated " << endl;
}

template <typename DataType>
SAMCreator<DataType>::SAMCreator(GlobalControlChannel& _control, unsigned int _channel_count,
                                 unsigned int _length, unsigned int _width, sc_trace_file* tf,
                                 uint16_t _start_uid, uint16_t _end_uid)
    : control(_control),
      channel_count(_channel_count),
      length(_length),
      width(_width),
      tf(tf),
      start_uid(_start_uid),
      end_uid(_end_uid) {}

template <typename DataType>
SAM<DataType>* SAMCreator<DataType>::operator()(const char* name, size_t) {
  return new SAM<DataType>(name, control, channel_count, length, width, tf, start_uid, end_uid);
}

template struct SAMDataPortCreator<sc_int<32>>;
template struct SAM<sc_int<32>>;
template struct SAMCreator<sc_int<32>>;