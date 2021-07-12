#if !defined(__SAM_CPP__)
#define __SAM_CPP__

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <systemc>

#include "GlobalControl.hh"
#include "Memory.hh"

using std::cout;
using std::endl;
using std::string;
using namespace sc_core;
using namespace sc_dt;

// Forward declarations
template <typename DataType>
struct AddressGenerator;
template <typename DataType>
struct MemoryChannel;

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
  sc_signal<DataType> program_data;

 public:
  sc_port<GlobalControlChannel_IF> control;
  Memory<DataType> mem;
  sc_vector<AddressGenerator<DataType>> generators;
  sc_vector<MemoryChannel<DataType>> channels;
  sc_vector<sc_signal<bool>> channel_dma_periph_ready_valid;
  sc_vector<sc_vector<sc_out<DataType>>> read_channel_data;
  sc_vector<sc_vector<sc_in<DataType>>> write_channel_data;
  sc_in<DataType> program_in;
  sc_out<DataType> program_out;
  const unsigned int length, width, channel_count;

  void update();

  void in_port_propogate();

  void out_port_propogate();

  SAM(sc_module_name name, GlobalControlChannel& _control, unsigned int _channel_count,
      unsigned int _length, unsigned int _width, sc_trace_file* tf, uint16_t& _current_uid,
      uint16_t _end_uid);

  SC_HAS_PROCESS(SAM);
};

template <typename DataType>
struct SAMCreator {
  SAMCreator(GlobalControlChannel& _control, unsigned int _channel_count, unsigned int _length,
             unsigned int _width, sc_trace_file* tf, uint16_t _start_uid, uint16_t _end_uid);
  SAM<DataType>* operator()(const char* name, size_t);
  GlobalControlChannel& control;
  unsigned int channel_count;
  unsigned int length;
  unsigned int width;
  sc_trace_file* tf;
  uint16_t current_uid;  //! Main UID distributor; NOTE: You should not instantiate SAMCreators with
                         //! overlapping UID ranges to avoid collisions
  uint16_t end_uid;      //! Last UID this SAMCreator can assign
};

#endif
