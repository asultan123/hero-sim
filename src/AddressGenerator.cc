#include "AddressGenerator.hh"

Descriptor_2D::Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state,
                             unsigned int _x_count, int _x_modify, unsigned int _y_count,
                             int _y_modify)
{
    this->next = _next;
    this->start = _start;
    this->state = _state;
    this->x_count = _x_count;
    this->x_modify = _x_modify;
    this->y_count = _y_count;
    this->y_modify = _y_modify;
    this->x_counter = _x_count;
    this->y_counter = _y_count;
    this->repeat = 0;
}

Descriptor_2D::Descriptor_2D(const Descriptor_2D &rhs)
{
    this->next = rhs.next;
    this->start = rhs.start;
    this->state = rhs.state;
    this->x_count = rhs.x_count;
    this->x_modify = rhs.x_modify;
    this->y_count = rhs.y_count;
    this->y_modify = rhs.y_modify;
    this->x_counter = rhs.x_counter;
    this->y_counter = rhs.y_counter;
    this->repeat = rhs.repeat;
}

Descriptor_2D Descriptor_2D::default_descriptor()
{
    return {0, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0};
}

void Descriptor_2D::x_count_update(int count)
{
    this->x_count = count;
    this->x_counter = count;
}

void Descriptor_2D::y_count_update(int count)
{
    this->y_count = count;
    this->y_counter = count;
}

bool Descriptor_2D::operator==(const Descriptor_2D &rhs)
{
    return this->next == rhs.next && this->start == rhs.start &&
           this->state == rhs.state && this->x_count == rhs.x_count &&
           this->x_modify == rhs.x_modify && this->y_count == rhs.y_count &&
           this->y_modify == rhs.y_modify;
}

template <typename DataType>
void AddressGenerator<DataType>::resetIndexingCounters()
{
    x_count_remaining = descriptors[execute_index].x_count;
    y_count_remaining = descriptors[execute_index].y_count;
}

template <typename DataType>
void AddressGenerator<DataType>::loadInternalCountersFromIndex(unsigned int index)
{
    current_ram_index = descriptors.at(index).start;
    x_count_remaining = descriptors.at(index).x_count;
    y_count_remaining = descriptors.at(index).y_count;
    repeat = descriptors.at(index).repeat;
}

template <typename DataType>
void AddressGenerator<DataType>::loadProgram(const vector<Descriptor_2D> &newProgram)
{
    descriptors.clear();
    copy(newProgram.begin(), newProgram.end(), std::back_inserter(descriptors));
}

template <typename DataType>
void AddressGenerator<DataType>::resetProgramMemory()
{
    execute_index = 0;
    descriptors.clear();
    descriptors.push_back(Descriptor_2D::default_descriptor());
}

template <typename DataType>
Descriptor_2D AddressGenerator<DataType>::currentDescriptor()
{
    return descriptors.at(execute_index);
}

template <typename DataType>
Descriptor_2D AddressGenerator<DataType>::nextDescriptor()
{
    return descriptors.at(descriptors[execute_index].next);
}

template <typename DataType>
void AddressGenerator<DataType>::RGENWAITupdateCurrentIndex()
{
    // TODO: NOT IMPLEMENTED!
    assert(0);
    // if (x_count_remaining != 0)
    // {
    //     x_count_remaining = x_count_remaining - 1;
    // }

    // if (x_count_remaining == 0)
    // {
    //     if (y_count_remaining != 0)
    //     {
    //         current_ram_index = current_ram_index + currentDescriptor().y_modify;
    //         // HACK WITH CHANNEL->SET_ADDR
    //         channel->set_addr(current_ram_index + currentDescriptor().y_modify);
    //         x_count_remaining = currentDescriptor().x_count;
    //         y_count_remaining = y_count_remaining - 1;
    //     }
    // }
    // else
    // {
    //     // HACK WITH CHANNEL->SET_ADDR
    //     current_ram_index = current_ram_index + currentDescriptor().x_modify;
    //     channel->set_addr(current_ram_index + currentDescriptor().x_modify);
    // }
}


template <typename DataType>
void AddressGenerator<DataType>::updateCurrentIndex()
{
    if (x_count_remaining != 0)
    {
        x_count_remaining = x_count_remaining - 1;
    }

    if (x_count_remaining == 0)
    {
        if (y_count_remaining != 0)
        {
            current_ram_index = current_ram_index + currentDescriptor().y_modify;
            // HACK WITH CHANNEL->SET_ADDR
            channel->set_addr(current_ram_index + currentDescriptor().y_modify);
            x_count_remaining = currentDescriptor().x_count;
            y_count_remaining = y_count_remaining - 1;
        }
    }
    else
    {
        // HACK WITH CHANNEL->SET_ADDR
        current_ram_index = current_ram_index + currentDescriptor().x_modify;
        channel->set_addr(current_ram_index + currentDescriptor().x_modify);
    }
}

template <typename DataType>
bool AddressGenerator<DataType>::descriptorComplete()
{
    return (x_count_remaining == 0 && y_count_remaining == 0 && repeat == 0);
}

template <typename DataType>
void AddressGenerator<DataType>::loadNextDescriptor()
{
    execute_index = currentDescriptor().next;
    loadInternalCountersFromIndex(currentDescriptor().next);
    channel->set_addr(nextDescriptor().start);
}

template <typename DataType>
void AddressGenerator<DataType>::update()
{
    if (control->reset())
    {
        resetProgramMemory();
        loadInternalCountersFromIndex(0);
        programmed = false;
        first_cycle = false;
        channel->reset();
        std::cout << "@ " << sc_time_stamp() << " " << this->name()
                  << ":MODULE has been reset" << std::endl;
    }
    else if (control->program())
    {
        // TODO: Extend with programming logic
        execute_index = 0;
        loadInternalCountersFromIndex(0);
        channel->set_addr(descriptors.at(0).start);
        programmed = true;
        first_cycle = true;
        std::cout << "@ " << sc_time_stamp() << " " << this->name()
                  << ":MODULE has been programmed" << std::endl;
    }
    else if (control->enable() && programmed)
    {
        // Update internal address counters, ignore for first cycle due to channel enable delay
        if (!first_cycle && (currentDescriptor().state == DescriptorState::GENERATE ||
                             currentDescriptor().state == DescriptorState::WAIT ||
                             currentDescriptor().state == DescriptorState::RGENWAIT))
        {
            if(currentDescriptor().state == DescriptorState::RGENWAIT)
            {
                RGENWAITupdateCurrentIndex();
            }
            else
            {
                updateCurrentIndex();
            }
            
            if (descriptorComplete())
            {
                loadNextDescriptor();
                channel->set_enable(false);
                first_cycle = true;
            }
        }
        else
        {
            first_cycle = false;
        }

        if (!descriptorComplete())
        {
            switch (currentDescriptor().state)
            {
            case DescriptorState::GENERATE:
            {
                channel->set_enable(true);
                break;
            }
            case DescriptorState::WAIT:
            case DescriptorState::SUSPENDED:
            {
                channel->set_enable(false);
                break;
            }
            default:
            {
                std::cout << "@ " << sc_time_stamp() << " " << this->name()
                          << ": Is in an invalid state! ... exitting" << std::endl;
                exit(-1);
            }
            }
        }

        // else
        // {
        //     // pre enable for next descriptors first cycle
        //     switch (nextDescriptor().state)
        //     {
        //     case DescriptorState::GENERATE:
        //     {
        //         channel->set_enable(true);
        //         break;
        //     }
        //     case DescriptorState::WAIT:
        //     case DescriptorState::SUSPENDED:
        //     {
        //         channel->set_enable(false);
        //         break;
        //     }
        //     default:
        //     {
        //         std::cout << "@ " << sc_time_stamp() << " " << this->name()
        //                 << ": Is in an invalid state! ... exitting" << std::endl;
        //         exit(-1);
        //     }
        //     }
        // }
    }
}

// Constructor
template <typename DataType>
AddressGenerator<DataType>::AddressGenerator(sc_module_name name, GlobalControlChannel &_control,
                                             sc_trace_file *_tf)
    : sc_module(name), control("control"), channel("channel"), tf(_tf),
      execute_index("execute_index"),
      current_ram_index("current_ram_index"),
      x_count_remaining("x_count_remaining"),
      y_count_remaining("y_count_remaining"),
      repeat("repeat")
{
    control(_control);
    _clk(control->clk());
    _reset(control->reset());
    execute_index = 0;
// sc_trace(tf, this->execute_index, (this->execute_index.name()));
    sc_trace(tf, this->execute_index, (this->execute_index.name()));
    sc_trace(tf, this->current_ram_index, (this->current_ram_index.name()));
    sc_trace(tf, this->x_count_remaining, (this->x_count_remaining.name()));
    sc_trace(tf, this->y_count_remaining, (this->y_count_remaining.name()));
    sc_trace(tf, this->repeat, (this->repeat.name()));

    SC_METHOD(update);
    sensitive << _clk.pos();
    sensitive << _reset.pos();

    // connect signals
    std::cout << "ADDRESS_GENERATOR MODULE: " << name
              << " has been instantiated " << std::endl;
}

template <typename DataType>
AddressGeneratorCreator<DataType>::AddressGeneratorCreator(GlobalControlChannel &_control, sc_trace_file *_tf) : tf(_tf), control(_control) {}

template <typename DataType>
AddressGenerator<DataType> *AddressGeneratorCreator<DataType>::operator()(const char *name, size_t)
{
    return new AddressGenerator<DataType>(name, control, tf);
}

template struct AddressGenerator<sc_int<32>>;
template struct AddressGeneratorCreator<sc_int<32>>;
