#if !defined(__LAYER_GENERATION_CC__)
#define __LAYER_GENERATION_CC__

#include <xtensor/xarray.hpp>
#include "hero.hh"
#include <xtensor/xpad.hpp>
#include <deque>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <stdexcept>

#define PAD -1

using std::deque;
using std::vector;

namespace LayerGeneration
{
    // TODO: enable layer generation with arbitrary DataTypes other than int
    template <typename DataType>
    xt::xarray<int> generate_ifmap(Hero::Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w);

    template <typename DataType>
    xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel);

    template <typename DataType>
    xt::xarray<int> pad_weights(Hero::Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel);
    
    template <typename DataType>
    bool validate_output(Hero::Arch<DataType> &arch, xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<int> arch_output);
    bool validate_output_1x1(xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<int> arch_output);
}

#include "../src/layer_generation.cc"

#endif