#ifdef __INTELLISENSE__
#include "../include/Memory.hh"
#endif

template <typename DataType>
MemoryRowCreator<DataType>::MemoryRowCreator(unsigned int _width, sc_trace_file *_tf) : tf(_tf), width(_width)
{
}

template <typename DataType>
sc_vector<sc_signal<DataType>> *MemoryRowCreator<DataType>::operator()(const char *name, size_t)
{
    return new sc_vector<sc_signal<DataType>>(name, width);
}

template <typename DataType> void Memory<DataType>::update()
{
    if (control->reset())
    {
        access_counter = 0;
        for (auto &row : ram)
        {
            for (auto &col : row)
            {
                col = 0;
            }
        }
        std::cout << "@ " << sc_time_stamp() << " " << this->name() << ":MODULE has been reset" << std::endl;
    }
    else if (control->enable())
    {
        string comp_name = name();

        for (unsigned int channel_idx = 0; channel_idx < channel_count; channel_idx++)
        {
            if (channels[channel_idx]->enabled())
            {
                switch (channels[channel_idx]->mode())
                {
                case MemoryChannelMode::WRITE:
                    assert(channels[channel_idx]->get_width() == width);
                    for (unsigned int i = 0; i < width; i++)
                    {
                        access_counter++;
                        ram.at(channels[channel_idx]->addr()).at(i) = channels[channel_idx]->mem_read_data().at(i);
                    }
                    break;
                case MemoryChannelMode::READ:
                    assert(channels[channel_idx]->get_width() == width);
                    access_counter++;
                    channels[channel_idx]->mem_write_data(ram.at(channels[channel_idx]->addr()));
                    break;
                }
            }
            else
            {
                channels[channel_idx]->mem_write_data(0);
            }
        }
    }
}

template <typename DataType> void Memory<DataType>::print_memory_contents()
{
    for (const auto &row : ram)
    {
        for (const auto &col : row)
        {
            cout << col << " ";
        }
        cout << endl;
    }
}

// Constructor
template <typename DataType>
Memory<DataType>::Memory(sc_module_name name, GlobalControlChannel &_control, unsigned int _channel_count,
                         unsigned int _length, unsigned int _width, sc_trace_file *tf, bool trace_mem)
    : sc_module(name), ram("ram", _length, MemoryRowCreator<DataType>(_width, tf)), control("control"),
      channels("channel", _channel_count), width(_width), length(_length), channel_count(_channel_count),
      access_counter(0)

{
    if (trace_mem)
    {
        for (unsigned int row = 0; row < length; row++)
        {
            for (unsigned int col = 0; col < width; col++)
            {
                #ifdef DEBUG
                sc_trace(tf, ram[row][col], ram[row][col].name());
                #endif // DEBUG
            }
        }
    }

    control(_control);
    _clk(control->clk());

    SC_METHOD(update);

    sensitive << _clk.pos();
    sensitive << control->reset();

    cout << "MEMORY MODULE: " << name << " has been instantiated " << endl;
}

// template struct Memory<sc_int<32>>;
// template struct MemoryRowCreator<sc_int<32>>;