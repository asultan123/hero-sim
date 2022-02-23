#if !defined(__LAYER_GENERATION_CC__)
#define __LAYER_GENERATION_CC__

#include <xtensor/xarray.hpp>
#include "hero.hh"

namespace LayerGeneration{
    template <typename DataType>
    xt::xarray<int> generate_ifmap(Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w);

    template <typename DataType>
    xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel);

}


#endif 