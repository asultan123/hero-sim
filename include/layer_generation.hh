#if !defined(__LAYER_GENERATION_CC__)
#define __LAYER_GENERATION_CC__

#include <xtensor/xarray.hpp>
#include "hero.hh"
#include <xtensor/xpad.hpp>
#include <deque>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define PAD -1

using std::deque;
using std::vector;

namespace LayerGeneration
{

    enum UnrollOrientation
    {
        HORIZONTAL = 1,
        VERTICLE = 2
    };

    template <typename DataType>
    xt::xarray<int> generate_ifmap(Hero::Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w);

    template <typename DataType>
    xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel);

    template <typename DataType>
    xt::xarray<int> pad_weights(Hero::Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation);

    bool validate_output_1x1(xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<int> arch_output);
}

#include "../src/layer_generation.cc"

#endif