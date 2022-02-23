#include "layer_generation.hh"

namespace LayerGeneration{

template <typename DataType>
xt::xarray<int> generate_ifmap(Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w)
{
    auto input_size = ifmap_h * ifmap_w * channel_in;
    assert(input_size <= arch.ifmap_mem_size);

    xt::xarray<int> ifmap = xt::arange((int)1, input_size + 1);
    ifmap.reshape({channel_in, ifmap_h, ifmap_w});

    return ifmap;
}

template <typename DataType>
xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel)
{
    int kernel_size = kernel * kernel;
    xt::xarray<int> weights = xt::arange(1, channel_in_dim * filter_out_dim * kernel_size + 1);

    weights.reshape({filter_out_dim, channel_in_dim, kernel, kernel});
    return weights;
}

template <typename DataType>
xt::xarray<int> pad_weights(Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation)
{
    int kernel_size = kernel * kernel;
    
    vector<vector<deque<int>>> pe_weights(arch.filter_count, vector<deque<int>>(arch.channel_count, deque<int>()));

    long unsigned int verticle_padding;
    long unsigned int horizontal_padding;

    switch (unroll_orientation)
    {
    case UnrollOrientation::HORIZONTAL:
    {
        weights.reshape({filter_out_dim, channel_in_dim * kernel_size});
        verticle_padding = ceil((float)filter_out_dim / arch.filter_count) * arch.filter_count - filter_out_dim;
        horizontal_padding = ceil((float)(channel_in_dim * kernel_size) / arch.channel_count) * arch.channel_count - (channel_in_dim * kernel_size);
        break;
    }
    default:
        cout << "INVALID ORIENTATION" << endl;
        exit(EXIT_FAILURE);
        break;
    }

    xt::xarray<int> padded_weights = xt::pad(weights, {{0, verticle_padding}, {0, horizontal_padding}}, xt::pad_mode::constant, PAD);
    
    return padded_weights;
}

template xt::xarray<int> generate_weights<sc_int<32>>(int filter_out_dim, int channel_in_dim, int kernel);
template xt::xarray<int> generate_ifmap<sc_int<32>>(Arch<sc_int<32>> &arch, int channel_in, int ifmap_h, int ifmap_w);
template xt::xarray<int> pad_weights<sc_int<32>>(Arch<sc_int<32>> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation);

}