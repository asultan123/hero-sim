#if !defined(__SAM_CPP__)
#define __SAM_CPP__

#include <assert.h>

#include <iostream>
#include <string>
#include <systemc>

#include "AddressGenerator.hh"
#include "GlobalControl.hh"

using std::cout;
using std::endl;
using std::string;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType>
struct SAMDataPortCreator {
  SAMDataPortCreator(unsigned int _width, sc_trace_file* _tf);
  sc_vector<DataType>* operator()(const char* name, size_t);
  sc_trace_file* tf;
  unsigned int width;
};

template <typename DataType>
using OutDataPortCreator = SAMDataPortCreator<sc_out<DataType>>;

template <typename DataType>
using InDataPortCreator = SAMDataPortCreator<sc_in<DataType>>;

template <typename DataType>
struct SAM : public sc_module {
  // Member Signals
 private:
  sc_in_clk _clk;

 public:
  sc_port<GlobalControlChannel_IF> control;
  Memory<DataType> mem;
  sc_vector<AddressGenerator<DataType>> generators;
  sc_vector<MemoryChannel<DataType>> channels;
  sc_vector<sc_vector<sc_out<DataType>>> read_channel_data;
  sc_vector<sc_vector<sc_in<bool>>> write_channel_dma_data_ready;
  sc_vector<sc_vector<sc_out<bool>>> write_channel_dma_assert_read;
  sc_vector<sc_vector<sc_in<DataType>>> write_channel_data;
  const unsigned int length, width, channel_count;

  void update();

  void in_port_propogate();

  void out_port_propogate();

  void in_port_deassert_read();

  SAM(sc_module_name name, GlobalControlChannel& _control,
      unsigned int _channel_count, unsigned int _length, unsigned int _width,
      sc_trace_file* tf);

  SC_HAS_PROCESS(SAM);
};
#endif
