#ifdef __INTELLISENSE__
#include "../include/Memory_Channel.hh"
#endif

template <typename DataType>
MemoryChannel<DataType>::MemoryChannel(sc_module_name name, unsigned int width, sc_trace_file *tf)
    : sc_module(name), read_channel_data("read_channel_data", width), write_channel_data("write_channel_data", width),
      channel_addr("addr"), channel_enabled("enabled"), channel_mode("mode"), channel_width(width)
{
    channel_addr = 0;
    channel_enabled = false;
    channel_mode = MemoryChannelMode::READ;

    for (unsigned int i = 0; i < channel_width; i++)
    {
        sc_trace(tf, this->read_channel_data[i], (string(this->read_channel_data[i].name())));
        sc_trace(tf, this->write_channel_data[i], (string(this->write_channel_data[i].name())));
    }

    sc_trace(tf, this->channel_addr, (string(this->channel_addr.name())));
    sc_trace(tf, this->channel_enabled, (string(this->channel_enabled.name())));
    sc_trace(tf, this->channel_mode, (string(this->channel_mode.name())));

    cout << "MEMORY_CHANNEL CHANNEL: " << name << " has been instantiated " << endl;
}

template <typename DataType>
const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &MemoryChannel<DataType>::mem_read_data()
{
    return write_channel_data;
}

template <typename DataType> void MemoryChannel<DataType>::mem_write_data(const sc_vector<sc_signal<DataType>> &_data)
{
    assert(_data.size() == channel_width);
    for (unsigned int i = 0; i < channel_width; i++)
    {
        read_channel_data.at(i) = _data.at(i);
    }
}

template <typename DataType> void MemoryChannel<DataType>::mem_write_data(int _data)
{
    for (unsigned int i = 0; i < channel_width; i++)
    {
        read_channel_data.at(i) = DataType(_data);
    }
}

template <typename DataType>
const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &MemoryChannel<DataType>::channel_read_data()
{
    return read_channel_data;
}

template <typename DataType> const DataType &MemoryChannel<DataType>::channel_read_data_element(unsigned int col)
{
    assert(col <= channel_width && col >= 0);
    return read_channel_data.at(col);
}

template <typename DataType>
sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &MemoryChannel<DataType>::get_channel_read_data_bus()
{
    return read_channel_data;
}

template <typename DataType>
sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &MemoryChannel<DataType>::get_channel_write_data_bus()
{
    return write_channel_data;
}

template <typename DataType>
void MemoryChannel<DataType>::channel_write_data(const sc_vector<sc_signal<DataType>> &_data)
{
    assert(_data.size() == channel_width);
    for (unsigned int i = 0; i < channel_width; i++)
    {
        write_channel_data[i] = _data[i];
    }
}

template <typename DataType> void MemoryChannel<DataType>::channel_write_data_element(DataType _data, unsigned int col)
{
    assert(col <= channel_width && col >= 0);
    write_channel_data[col] = _data;
}

template <typename DataType> unsigned int MemoryChannel<DataType>::addr()
{
    return channel_addr;
}

template <typename DataType> void MemoryChannel<DataType>::set_addr(unsigned int addr)
{
    channel_addr = addr;
}

template <typename DataType> void MemoryChannel<DataType>::set_enable(bool status)
{
    channel_enabled = status;
}

template <typename DataType> bool MemoryChannel<DataType>::enabled()
{
    return channel_enabled;
}

template <typename DataType> void MemoryChannel<DataType>::set_mode(MemoryChannelMode mode)
{
    channel_mode = mode;
    // channel_mode = (mode == MemoryChannelMode::READ)? 0 : 1;
}

template <typename DataType> MemoryChannelMode MemoryChannel<DataType>::mode()
{
    return (MemoryChannelMode)channel_mode.read();
    // return (channel_mode.read() == 0)? MemoryChannelMode::READ : MemoryChannelMode::WRITE;
}

template <typename DataType> void MemoryChannel<DataType>::reset()
{
    for (auto &data : read_channel_data)
    {
        data = 0;
    }
    for (auto &data : write_channel_data)
    {
        data = 0;
    }
    channel_addr = 0;
    channel_enabled = false;
    channel_mode = MemoryChannelMode::READ;
    std::cout << "@ " << sc_time_stamp() << " " << this->name() << ": channel has been reset" << std::endl;
}

template <typename DataType> const unsigned int &MemoryChannel<DataType>::get_width()
{
    return channel_width;
}

template <typename DataType> void MemoryChannel<DataType>::register_port(sc_port_base &port_, const char *if_typename_)
{
    cout << "now binding    " << port_.name() << " to "
         << "interface: " << if_typename_ << " with channel width " << channel_width << endl;
}

template <typename DataType>
MemoryChannelCreator<DataType>::MemoryChannelCreator(unsigned int _width, sc_trace_file *_tf) : tf(_tf), width(_width)
{
}

template <typename DataType>
MemoryChannel<DataType> *MemoryChannelCreator<DataType>::operator()(const char *name, size_t)
{
    return new MemoryChannel<DataType>(name, width, tf);
}

// template struct MemoryChannel<sc_int<32>>;
// template struct MemoryChannelCreator<sc_int<32>>;
