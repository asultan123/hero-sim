#if !defined(__LAYER_GENERATION_CC__)
#define __LAYER_GENERATION_CC__

#include <xtensor/xarray.hpp>
#include "hero.hh"
#include <xtensor/xpad.hpp>
#include <deque>

#define PAD -1

using std::deque;
using std::vector;

namespace LayerGeneration{
    enum UnrollOrientation
    {
        HORIZONTAL = 1,
        VERTICLE = 2
    };

    template <typename DataType>
    xt::xarray<int> generate_ifmap(Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w);

    template <typename DataType>
    xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel);

    template <typename DataType>
    xt::xarray<int> pad_weights(Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation);
}


#endif 