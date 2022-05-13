#if !defined(__LAYER_GENERATION_CC__)
#define __LAYER_GENERATION_CC__

#include "hero.hh"
#include <deque>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>

#define PAD -1

using std::deque;
using std::vector;

namespace LayerGeneration
{
// TODO: #28 enable layer generation with arbitrary DataTypes other than int
template <typename DataType>
xt::xarray<int> generate_ifmap(Hero::Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w);

template <typename DataType> xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel);

template <typename DataType>
xt::xarray<int> pad_weights(Hero::Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim,
                            int kernel);

template <typename DataType>
bool validate_output(xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<DataType> arch_output);

} // namespace LayerGeneration

#include "../src/layer_generation.cc"

#endif