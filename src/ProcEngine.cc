#ifdef __INTELLISENSE__
#include "../include/ProcEngine.hh"
#endif

template <typename DataType>
PE<DataType>::PE(sc_module_name name, sc_trace_file *_tf)
    : sc_module(name), tf(_tf), psum_in("psum_in"), current_weight("weight")
{
    this->resetWeightIdx();
    this->resetWeights();
    this->programmed = false;
    this->weight_access_counter = 0;
    sc_trace(tf, this->psum_in, (this->psum_in.name()));
    sc_trace(tf, this->current_weight, (this->current_weight.name()));
}

template <typename DataType> DataType PE<DataType>::compute(sc_signal<DataType> &input)
{
    return this->current_weight.read() * input.read() + this->psum_in.read();
}

template <typename DataType> DataType PE<DataType>::compute(unsigned long int input)
{
    return this->current_weight.read() * input + this->psum_in.read();
}

template <typename DataType> void PE<DataType>::reset()
{
    this->resetWeightIdx();
    this->resetWeights();
    this->programmed = false;
}

template <typename DataType> void PE<DataType>::resetWeightIdx()
{
    this->weight_idx = 0;
}

template <typename DataType> void PE<DataType>::resetWeights()
{
    this->weights.clear();
}

template <typename DataType> void PE<DataType>::loadWeights(vector<int> &weights)
{
    this->weights = weights;
    weight_access_counter += weights.size();
}

template <typename DataType> void PE<DataType>::loadProgram(vector<Descriptor_2D> &_program)
{
    this->program.clear();
    for (auto &desc : _program)
    {
        this->program.push_back(desc);
    }
    this->prog_idx = 0;
    this->programmed = true;
    weight_access_counter += 1;
}

template <typename DataType> void PE<DataType>::updateState()
{
    if (this->programmed)
    {
        Descriptor_2D &current_desc = this->program.at(prog_idx);

        if (current_desc.state == DescriptorState::GENHOLD)
        {
            this->current_weight = this->weights[weight_idx];
            current_desc.x_counter--;
            if (current_desc.x_counter < 0)
            {
                current_desc.x_counter = current_desc.x_count;
                current_desc.y_counter--;
                weight_idx += current_desc.y_modify;
                weight_access_counter += 1;
            }
            if (current_desc.y_counter < 0)
            {
                this->prog_idx++;
            }
        }
        else if (current_desc.state == DescriptorState::WAIT)
        {
            current_desc.x_counter--;
            if (current_desc.x_counter < 0)
            {
                this->prog_idx++;
            }
        }
        else if (current_desc.state == DescriptorState::SUSPENDED)
        {
            return;
        }
        else
        {
            cout << "ERROR: Invalid Descriptor in pe program" << endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        cout << "ERROR: Attempted to update pe state without it being programmed" << endl;
        exit(EXIT_FAILURE);
    }
}

// template struct PE<int>;
// template struct PE<sc_int<32>>;
