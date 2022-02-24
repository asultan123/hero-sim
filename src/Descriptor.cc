#ifdef __INTELLISENSE__
#include "../include/Descriptor.hh"
#endif

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


void Descriptor_2D::make_sequential(vector<Descriptor_2D>& program)
{
    int idx = 1;
    for(auto& desc : program)
    {
        desc.next = idx++;
    }
    idx -= 2;
    program.at(program.size()-1).next = idx;
}

Descriptor_2D Descriptor_2D::delay_inst(int delay_time)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ 0,
        /*state*/ DescriptorState::WAIT,
        /*x_count*/ delay_time,
        /*x_modify*/ 0,
        /*y_count*/ 0,
        /*y_modify*/ 0);
}

Descriptor_2D Descriptor_2D::stream_inst(int start_idx, int stream_size, int repeats)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ start_idx,
        /*state*/ DescriptorState::GENERATE,
        /*x_count*/ stream_size,
        /*x_modify*/ 1,
        /*y_count*/ repeats,
        /*y_modify*/ -(stream_size));
}


Descriptor_2D Descriptor_2D::genhold_inst(int start_idx, int hold_time, int repeats, int access_offset)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ start_idx,
        /*state*/ DescriptorState::GENHOLD,
        /*x_count*/ hold_time,
        /*x_modify*/ 1,
        /*y_count*/ repeats,
        /*y_modify*/ access_offset);
}

Descriptor_2D Descriptor_2D::suspend_inst()
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
