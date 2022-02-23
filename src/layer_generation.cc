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

template xt::xarray<int> generate_weights<sc_int<32>>(int filter_out_dim, int channel_in_dim, int kernel);
template xt::xarray<int> generate_ifmap<sc_int<32>>(Arch<sc_int<32>> &arch, int channel_in, int ifmap_h, int ifmap_w);

}