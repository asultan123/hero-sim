#if !defined(__HERO_1x1__)
#define __HERO_1x1__

#include "AddressGenerator.hh"
#include "GlobalControl.hh"
#include "ProcEngine.hh"
#include "SAM.hh"
#include "SSM.hh"
#include <stdexcept>
#include <systemc.h>

namespace Hero
{
enum OperationMode
{
    RUN_1x1 = 1,
    RUN_3x3 = 2
};

enum KernelMapping
{
    HORIZONTAL = 1,
    VERTICLE = 2
};

template <typename DataType> struct SignalVectorCreator
{
    sc_trace_file *tf;
    unsigned int width;

    SignalVectorCreator(unsigned int _width, sc_trace_file *_tf);

    sc_vector<sc_signal<DataType>> *operator()(const char *name, size_t);
};

template <typename DataType> struct SAMVectorCreator
{
    GlobalControlChannel &control;
    unsigned int channel_count;
    unsigned int length;
    unsigned int width;
    sc_trace_file *tf;

    SAMVectorCreator(GlobalControlChannel &_control, unsigned int _channel_count, unsigned int _length,
                     unsigned int _width, sc_trace_file *_tf);

    SAM<DataType> *operator()(const char *name, size_t);
};

template <typename DataType> struct SSMVectorCreator
{
    GlobalControlChannel &control;
    unsigned int input_count;
    unsigned int output_count;
    sc_trace_file *tf;

    SSMVectorCreator(GlobalControlChannel &_control, unsigned int input_count, unsigned int output_count,
                     sc_trace_file *_tf);

    SSM<DataType> *operator()(const char *name, size_t);
};

template <typename DataType> struct PeCreator
{
    sc_trace_file *tf;
    PeCreator(sc_trace_file *_tf);
    PE<DataType> *operator()(const char *name, size_t);
};

template <typename DataType> struct Arch : public sc_module
{
    // Member Signals
  private:
    sc_in_clk _clk;

  public:
    sc_port<GlobalControlChannel_IF> control;
    sc_vector<PE<DataType>> pe_array;
    sc_trace_file *tf;
    SAM<DataType> psum_mem;
    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_read;

    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_write;
    SAM<DataType> ifmap_mem;
    sc_vector<SAM<DataType>> ifmap_reuse_chain;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_reuse_chain_signals;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_read;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_write;
    sc_vector<SSM<DataType>> ssm;

    unsigned int dram_access_counter{0};
    int filter_count;
    int channel_count;
    int psum_mem_size;
    int ifmap_mem_size;

    KernelMapping kmapping;
    OperationMode mode;

    void suspend_monitor();

    void update_1x1();
    void update_3x3();

    void set_channel_modes();

    // Constructor
    Arch(sc_module_name name, GlobalControlChannel &_control, int filter_count, int channel_count, int psum_mem_size,
         int ifmap_mem_size, sc_trace_file *_tf, KernelMapping kmapping = KernelMapping::HORIZONTAL,
         OperationMode mode = OperationMode::RUN_1x1);

    SC_HAS_PROCESS(Arch);
};
} // namespace Hero

#include "../src/hero.cc"

#endif