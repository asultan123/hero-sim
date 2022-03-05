#ifdef __INTELLISENSE__
#include "../include/hero.hh"
#endif

namespace Hero
{
    template <typename DataType>
    SignalVectorCreator<DataType>::SignalVectorCreator(unsigned int _width, sc_trace_file *_tf) : tf(_tf), width(_width){};

    template <typename DataType>
    sc_vector<sc_signal<DataType>> *SignalVectorCreator<DataType>::operator()(const char *name, size_t)
    {
        return new sc_vector<sc_signal<DataType>>(name, width);
    }

    template <typename DataType>
    PeCreator<DataType>::PeCreator(sc_trace_file *_tf) : tf(_tf) {}

    template <typename DataType>
    PE<DataType> *PeCreator<DataType>::operator()(const char *name, size_t)
    {
        return new PE<DataType>(name, this->tf);
    }

    template <typename DataType>
    SAMVectorCreator<DataType>::SAMVectorCreator(GlobalControlChannel &_control, unsigned int _channel_count, unsigned int _length, unsigned int _width, sc_trace_file *_tf) : control(_control), channel_count(_channel_count), length(_length), width(_width), tf(_tf) {}

    template <typename DataType>
    SAM<DataType> *SAMVectorCreator<DataType>::operator()(const char *name, size_t)
    {
        return new SAM<DataType>(name, control, channel_count, length, width, tf);
    }

    template <typename DataType>
    void Arch<DataType>::suspend_monitor()
    {
        while (1)
        {
            while (control->enable())
            {
                bool pes_suspended = true;
                for (auto &pe : pe_array)
                {
                    pes_suspended &= (pe.program.at(pe.prog_idx).state == DescriptorState::SUSPENDED);
                }
                bool ifmap_generators_suspended = true;
                for (auto &gen : ifmap_mem.generators)
                {
                    ifmap_generators_suspended &= (gen.currentDescriptor().state == DescriptorState::SUSPENDED);
                }
                bool psum_generators_suspended = true;
                for (auto &gen : psum_mem.generators)
                {
                    psum_generators_suspended &= (gen.currentDescriptor().state == DescriptorState::SUSPENDED);
                }
                if (pes_suspended && ifmap_generators_suspended && psum_generators_suspended)
                {
                    sc_stop();
                }
                wait();
            }
            wait();
        }
    }

    template <typename DataType>
    void Arch<DataType>::update_3x3()
    {
        while (1)
        {
            while (control->enable())
            {
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    PE<DataType> &first_pe_in_row = this->pe_array[filter_row * channel_count];
                    first_pe_in_row.psum_in = psum_mem_read.at(filter_row + filter_count).at(0).read();
                }
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (int channel_column = 0; channel_column < channel_count - 1; channel_column++)
                    {
                        // int channel_group = channel_column / 3;
                        PE<DataType> &cur_pe = this->pe_array[filter_row * channel_count + channel_column];
                        PE<DataType> &next_pe = this->pe_array[filter_row * channel_count + channel_column + 1];
                        if (cur_pe.current_weight.read() != -1)
                        {
                            cur_pe.active_counter++;
                            next_pe.psum_in = cur_pe.compute(ifmap_mem_read[channel_column][0].read());
                        }
                        else
                        {
                            // bypass
                            cur_pe.inactive_counter++;
                            next_pe.psum_in = cur_pe.psum_in.read();
                        }
                        cur_pe.updateState();
                    }
                    PE<DataType> &last_pe = this->pe_array[filter_row * channel_count + channel_count - 1];
                    if (last_pe.current_weight.read() != -1)
                    {
                        last_pe.active_counter++;
                        psum_mem_write[filter_row][0] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    }
                    else
                    {
                        last_pe.inactive_counter++;
                        psum_mem_write[filter_row][0] = last_pe.psum_in.read();
                    }
                    last_pe.updateState();
                }
                wait();
            }
            wait();
        }
    }

    template <typename DataType>
    void Arch<DataType>::update_1x1()
    {
        while (1)
        {
            while (control->enable())
            {
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    PE<DataType> &first_pe_in_row = this->pe_array[filter_row * channel_count];
                    first_pe_in_row.psum_in = psum_mem_read.at(filter_row + filter_count).at(0).read();
                }
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (int channel_column = 0; channel_column < channel_count - 1; channel_column++)
                    {
                        PE<DataType> &cur_pe = this->pe_array[filter_row * channel_count + channel_column];
                        PE<DataType> &next_pe = this->pe_array[filter_row * channel_count + channel_column + 1];
                        if (cur_pe.current_weight.read() != -1)
                        {
                            cur_pe.active_counter++;
                            next_pe.psum_in = cur_pe.compute(ifmap_mem_read[channel_column][0].read());
                        }
                        else
                        {
                            // bypass
                            cur_pe.inactive_counter++;
                            next_pe.psum_in = cur_pe.psum_in.read();
                        }
                        cur_pe.updateState();
                    }
                    PE<DataType> &last_pe = this->pe_array[filter_row * channel_count + channel_count - 1];
                    if (last_pe.current_weight.read() != -1)
                    {
                        last_pe.active_counter++;
                        psum_mem_write[filter_row][0] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    }
                    else
                    {
                        last_pe.inactive_counter++;
                        psum_mem_write[filter_row][0] = last_pe.psum_in.read();
                    }
                    last_pe.updateState();
                }
                wait();
            }
            wait();
        }
    }

    template <typename DataType>
    void Arch<DataType>::set_channel_modes()
    {
        for (int i = 0; i < filter_count; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
        }
        for (int i = filter_count; i < filter_count * 2; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
        }

        for (int i = 0; i < channel_count; i++)
        {
            ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
        }
    }

    template <typename DataType>
    Arch<DataType>::Arch(
        sc_module_name name,
        GlobalControlChannel &_control,
        int filter_count,
        int channel_count,
        int psum_mem_size,
        int ifmap_mem_size,
        sc_trace_file *_tf,
        KernelMapping kmapping,
        OperationMode mode) : sc_module(name),
                        pe_array("pe_array", filter_count * channel_count, PeCreator<DataType>(_tf)),
                        tf(_tf),
                        psum_mem("psum_mem", _control, filter_count * 2, psum_mem_size, 1, _tf),
                        psum_mem_read("psum_mem_read", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                        psum_mem_write("psum_mem_write", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                        ifmap_mem("ifmap_mem", _control, channel_count, ifmap_mem_size, 1, _tf),
                        ifmap_reuse_chain("ifmap_reuse_chain"),
                        ifmap_mem_read("ifmap_mem_read", channel_count, SignalVectorCreator<DataType>(1, tf)),
                        ifmap_mem_write("ifmap_mem_write", channel_count, SignalVectorCreator<DataType>(1, tf)),
                        kmapping(kmapping),
                        mode(mode)
    {
        control(_control);
        _clk(control->clk());
        this->filter_count = filter_count;
        this->channel_count = channel_count;
        this->psum_mem_size = psum_mem_size;
        this->ifmap_mem_size = ifmap_mem_size;

        for (int i = 0; i < filter_count * 2; i++)
        {
            psum_mem.read_channel_data[i][0](psum_mem_read[i][0]);
            psum_mem.write_channel_data[i][0](psum_mem_write[i][0]);
        }
        for (int i = 0; i < filter_count; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
            sc_trace(tf, psum_mem_write[i][0], (this->psum_mem_write[i][0].name()));
        }
        for (int i = filter_count; i < filter_count * 2; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
            sc_trace(tf, psum_mem_read[i][0], (this->psum_mem_read[i][0].name()));
        }

        for (int i = 0; i < channel_count; i++)
        {
            ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
            ifmap_mem.read_channel_data[i][0](ifmap_mem_read[i][0]);
            ifmap_mem.write_channel_data[i][0](ifmap_mem_write[i][0]);
            sc_trace(tf, ifmap_mem_read[i][0], (this->ifmap_mem_read[i][0].name()));
        }

        if (mode == OperationMode::RUN_1x1)
        {
            SC_THREAD(update_1x1);
        }
        else if (mode == OperationMode::RUN_3x3)
        {
            assert(this->channel_count % 3 == 0);
            unsigned int kernel_groups_count = this->channel_count / 9;
            unsigned int total_sams_in_ifmap_chain = kernel_groups_count * 2;
            ifmap_reuse_chain.init(total_sams_in_ifmap_chain, SAMVectorCreator<DataType>(
                                                                  _control,
                                                                  2,   // port count
                                                                  512, // over estimating length
                                                                  1,   // width
                                                                  _tf));
            SC_THREAD(update_3x3);
        }
        else
        {
            throw std::invalid_argument("Recevied invalid hero arch operation mode");
        }

        sensitive << _clk.pos();
        sensitive << control->reset();

        SC_THREAD(suspend_monitor);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "Arch MODULE: " << name << " has been instantiated " << endl;
    }

    // template struct Arch<sc_int<32>>;
}