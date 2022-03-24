#ifdef __INTELLISENSE__
#include "../include/descriptor_compiler.hh"
#endif

namespace GenerateDescriptors
{

namespace
{

template <typename DataType>
xt::xarray<int> get_ifmap_mem_run_bitmap(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, horizontal_tile_count, (int)arch.channel_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1];
             channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count),
                                       xt::range(channel_offset, channel_offset + arch.channel_count));
            for (int channel = 0; channel < arch.channel_count; channel++)
            {
                int verticle_tile_idx = filter_offset / arch.filter_count;
                int horizontal_tile_idx = channel_offset / arch.channel_count;
                if (tiled_view(0, channel) != -1)
                {
                    run_bitmap(verticle_tile_idx, horizontal_tile_idx, channel) = 1;
                }
            }
        }
    }
    return run_bitmap;
}

template <typename DataType> void print_xtensor(xt::xarray<DataType> array)
{
    std::cout << array << endl;
}

} // namespace

namespace GenerateDescriptors1x1
{
template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
{
    int stream_size = ifmap_h * ifmap_w;
    int delay_offset = 1;
    for (int channel_column = 0; channel_column < arch.channel_count; channel_column++)
    {
        for (int filter_row = 0; filter_row < arch.filter_count; filter_row++)
        {
            PE<DataType> &cur_pe = arch.pe_array[filter_row * arch.channel_count + channel_column];
            vector<Descriptor_2D> program;
            program.push_back(Descriptor_2D::delay_inst(channel_column + delay_offset));
            program.push_back(Descriptor_2D::genhold_inst(0, stream_size, cur_pe.weights.size() - 1, 1));
            program.push_back(Descriptor_2D::suspend_inst());
            cur_pe.loadProgram(program);
        }
    }
}

template <typename DataType>
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    int stream_size = ofmap_h * ofmap_w;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, (int)arch.filter_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1];
             channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count),
                                       xt::range(channel_offset, channel_offset + arch.channel_count));
            for (int filter = 0; filter < arch.filter_count; filter++)
            {
                int verticle_tile_idx = filter_offset / arch.filter_count;
                if (tiled_view(filter, 0) != -1)
                {
                    run_bitmap(verticle_tile_idx, filter) = 1;
                }
            }
        }
    }

    // cout << padded_weights << endl;
    // cout << run_bitmap << endl;

    // TODO: #32
    for (int write_gen_idx = 0; write_gen_idx < arch.filter_count; write_gen_idx++)
    {
        vector<Descriptor_2D> program;

        program.push_back(Descriptor_2D::delay_inst(arch.channel_count + 1));
        for (int v = 0; v < verticle_tile_count; v++)
        {
            auto active = run_bitmap(v, write_gen_idx);
            if (active)
            {
                for (int h = 0; h < horizontal_tile_count; h++)
                {
                    program.push_back(Descriptor_2D::stream_inst(
                        v * arch.filter_count * stream_size + write_gen_idx * stream_size, stream_size - 1, 0));
                }
            }
        }

        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(write_gen_idx).loadProgram(program);
    }

    for (int read_gen_idx = arch.filter_count; read_gen_idx < arch.filter_count * 2; read_gen_idx++)
    {
        vector<Descriptor_2D> program;
        program.push_back(Descriptor_2D::delay_inst(3));

        for (int v = 0; v < verticle_tile_count; v++)
        {
            auto active = run_bitmap(v, (read_gen_idx - arch.filter_count));
            if (active)
            {
                program.push_back(Descriptor_2D::delay_inst(stream_size - 4 * (v == 0) - 1));
                for (int h = 1; h < horizontal_tile_count; h++)
                {
                    program.push_back(Descriptor_2D::stream_inst((v * arch.filter_count * stream_size) +
                                                                     (read_gen_idx - arch.filter_count) * stream_size,
                                                                 stream_size - 1, 0));
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(read_gen_idx).loadProgram(program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w)
{
    auto run_bitmap = get_ifmap_mem_run_bitmap(arch, padded_weights);
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;
    // cout << padded_weights << endl;

    // cout << run_bitmap << endl;
    // TODO: #32 Separate loading descriptors into hero arch from compilation and provide programming api in hero
    // implementation
    int ag_idx = 0;
    for (auto &ag : arch.ifmap_mem.generators)
    {
        std::deque<Descriptor_2D> program;
        auto systolic_delay = Descriptor_2D::delay_inst(ag_idx);
        program.push_back(systolic_delay);
        for (int v = 0; v < verticle_tile_count; v++)
        {
            for (int h = 0; h < horizontal_tile_count; h++)
            {
                int active = run_bitmap(v, h, ag_idx);
                int stream_size = ifmap_h * ifmap_w;
                int stream_start_idx = h * arch.channel_count * stream_size + ag_idx * stream_size;

                if (active)
                {
                    // TODO #42
                    auto stream_inst = Descriptor_2D::stream_inst(stream_start_idx, stream_size - 1, 0);
                    program.push_back(stream_inst);
                }
                else
                {
                    auto delay_inst = Descriptor_2D::delay_inst(stream_size - 1);
                    program.push_back(delay_inst);
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        vector<Descriptor_2D> prog_vec(program.begin(), program.end());
        Descriptor_2D::make_sequential(prog_vec);
        ag.loadProgram(prog_vec);
        ag_idx++;
    }
}

} // namespace GenerateDescriptors1x1

namespace GenerateDescriptors3x3
{

template <typename DataType> void generate_and_load_ssm_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
{
    // TODO: #41

    throw "Not implemented";
}

template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
{
    throw "Not implemented";
}

template <typename DataType>
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w)
{
    throw "Not implemented";
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w)
{
    throw "Not implemented";

    // auto run_bitmap = get_ifmap_mem_run_bitmap(arch, padded_weights);
    // int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    // int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;
    // int ag_idx = 0;
    // const int stream_size = ifmap_h * ifmap_w;
    // for (auto &ag : arch.ifmap_mem.generators)
    // {
    //     std::deque<Descriptor_2D> program;
    //     auto initialization_delay = Descriptor_2D::delay_inst(ag_idx * stream_size);
    //     program.push_back(initialization_delay);
    //     for (int v = 0; v < verticle_tile_count; v++)
    //     {
    //         for (int h = 0; h < horizontal_tile_count; h++)
    //         {
    //             int active = run_bitmap(v, h, ag_idx);
    //             int stream_start_idx = h * arch.channel_count * stream_size + ag_idx * stream_size;
    //             if (active)
    //             {
    //                 auto stream_inst = Descriptor_2D::stream_inst(stream_start_idx, stream_size - 1, 0);
    //                 program.push_back(stream_inst);
    //             }
    //             else
    //             {
    //                 auto delay_inst = Descriptor_2D::delay_inst(stream_size - 1);
    //                 program.push_back(delay_inst);
    //             }
    //             const int channel_memories_to_wait_for = (arch.channel_count - 1);
    //             auto delay_inst = Descriptor_2D::delay_inst(channel_memories_to_wait_for * (stream_size - 1));
    //             program.push_back(delay_inst);
    //         }
    //     }
    //     program.push_back(Descriptor_2D::suspend_inst());
    //     Descriptor_2D::make_sequential(program);
    //     ag.loadProgram(prog_vec);
    //     ag_idx++;
    // }
}

template <typename DataType>
void generate_and_load_ifmap_channel_to_reuse_chain_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights,
                                                            int ifmap_h, int ifmap_w)
{
    throw "Not implemented";
}

} // namespace GenerateDescriptors3x3

template <typename DataType>
void generate_and_load_arch_descriptors(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w,
                                        xt::xarray<int> padded_weights, int ofmap_h, int ofmap_w)
{

    switch (arch.mode)
    {
    case Hero::OperationMode::RUN_1x1:
        GenerateDescriptors1x1::generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors1x1::generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
        GenerateDescriptors1x1::generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);
        break;
    case Hero::OperationMode::RUN_3x3:
        GenerateDescriptors3x3::generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_ssm_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_ifmap_channel_to_reuse_chain_program(arch, padded_weights, ifmap_h,
                                                                                       ifmap_w);
        GenerateDescriptors3x3::generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);
        break;
    default:
        throw std::invalid_argument("Invalid architecture operation mode");
        break;
    }
}

} // namespace GenerateDescriptors
