#if !defined(__DESCRIPTOR_COMPILER_CC__)
#define __DESCRIPTOR_COMPILER_CC__

#include "hero.hh"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <vector>
#include <deque>

using std::deque;
using std::vector;
namespace GenerateDescriptors1x1
{
    template <typename DataType>
    void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w);

    template <typename DataType>
    void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h, int ofmap_w);

    template <typename DataType>
    void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h, int ifmap_w);

}

// namespace GenerateDescriptors3x3
// {

// }

#include "../src/descriptor_compiler.cc"

#endif