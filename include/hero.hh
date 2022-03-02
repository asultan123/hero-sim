#if !defined(__HERO_1x1__)
#define __HERO_1x1__

#include <systemc.h>
#include "ProcEngine.hh"
#include "SAM.hh"
#include "GlobalControl.hh"
#include "AddressGenerator.hh"

namespace Hero
{
    template <typename DataType>
    struct SignalVectorCreator
    {
        sc_trace_file *tf;
        unsigned int width;

        SignalVectorCreator(unsigned int _width, sc_trace_file *_tf);

        sc_vector<sc_signal<DataType>> *operator()(const char *name, size_t);
    };

    template <typename DataType>
    struct PeCreator
    {
        sc_trace_file *tf;
        PeCreator(sc_trace_file *_tf);
        PE<DataType> *operator()(const char *name, size_t);
    };

    template <typename DataType>
    struct Arch : public sc_module
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
        sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_read;
        sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_write;

        unsigned int dram_access_counter{0};
        int filter_count;
        int channel_count;
        int psum_mem_size;
        int ifmap_mem_size;

        void suspend_monitor();

        void update_1x1();
        void update_3x3();

        void set_channel_modes();

        // Constructor
        Arch(
            sc_module_name name,
            GlobalControlChannel &_control,
            int filter_count,
            int channel_count,
            int psum_mem_size,
            int ifmap_mem_size,
            sc_trace_file *_tf);

        SC_HAS_PROCESS(Arch);
    };
}

#include "../src/hero.cc"

#endif