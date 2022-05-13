#ifdef __INTELLISENSE__
#include "../include/layer_generation.hh"
#endif
namespace LayerGeneration
{
namespace
{
template <typename DataType>
xt::xarray<DataType> evaluate_expected_output(xt::xarray<DataType> ifmap, xt::xarray<DataType> weights)
{
    assert(weights.shape().size() == 4);
    assert(ifmap.shape().size() == 3);
    assert(weights.shape(3) == weights.shape(2));

    // ifmap channel = weights channel in
    assert(ifmap.shape(0) == weights.shape(1));
    int ifmap_w = ifmap.shape(2);
    int ifmap_h = ifmap.shape(1);
    int ifmap_c = ifmap.shape(0);

    int kernel = weights.shape(3);
    assert(ifmap_w >= kernel);
    assert(ifmap_h >= kernel);

    int ofmap_w = ifmap_w - (kernel - 1);
    int ofmap_h = ifmap_h - (kernel - 1);
    int ofmap_c = weights.shape(0);

    xt::xarray<DataType> ofmap = xt::arange(ofmap_c * ofmap_w * ofmap_h).reshape({ofmap_c, ofmap_h, ofmap_w});

    // conv2d stride 1
    for (auto f = 0; f < ofmap_c; f++)
    {
        auto weight_tensor_view = xt::view(weights, f, xt::all(), xt::all(), xt::all());
        xt::xarray<DataType> flatten_weight(xt::flatten(weight_tensor_view));
        for (auto h = 0; h < ofmap_h; h++)
        {
            for (auto w = 0; w < ofmap_w; w++)
            {
                auto ifmap_tensor_view = xt::view(ifmap, xt::all(), xt::range(h, h + kernel), xt::range(w, w + kernel));
                xt::xarray<DataType> flattened_ifmap(xt::flatten(ifmap_tensor_view));
                DataType sum = 0;
                for(auto ifmap_it = flattened_ifmap.begin(), weight_it = flatten_weight.begin(); ifmap_it != flattened_ifmap.end() && weight_it != flatten_weight.end(); ifmap_it++, weight_it++)
                {
                    sum += (*ifmap_it) * (*weight_it);

                }
                // xt::xarray<DataType> val = xt::linalg::dot(flattened_ifmap, flatten_weight);
                ofmap(f, h, w) = sum;
            }
        }
    }

    return ofmap;
}

} // namespace

template <typename DataType>
xt::xarray<int> generate_ifmap(Hero::Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w)
{
    auto input_size = ifmap_h * ifmap_w * channel_in;
    assert(input_size <= arch.ifmap_mem_size);

    xt::xarray<int> ifmap = xt::arange((int)1, input_size + 1);
    ifmap.reshape({channel_in, ifmap_h, ifmap_w});

    return ifmap;
}

template <typename DataType> xt::xarray<int> generate_weights(int filter_out_dim, int channel_in_dim, int kernel)
{
    int kernel_size = kernel * kernel;
    xt::xarray<int> weights = xt::arange(1, channel_in_dim * filter_out_dim * kernel_size + 1);

    weights.reshape({filter_out_dim, channel_in_dim, kernel, kernel});
    return weights;
}

template <typename DataType>
xt::xarray<int> pad_weights(Hero::Arch<DataType> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim,
                            int kernel)
{
    Hero::KernelMapping unroll_orientation = arch.kmapping;
    int kernel_size = kernel * kernel;

    vector<vector<deque<int>>> pe_weights(arch.filter_count, vector<deque<int>>(arch.channel_count, deque<int>()));

    long unsigned int verticle_padding;
    long unsigned int horizontal_padding;

    switch (unroll_orientation)
    {
    case Hero::KernelMapping::HORIZONTAL:
    {
        if (arch.channel_count % kernel_size != 0)
        {
            throw std::invalid_argument(
                "Architecture channel count has to be a multiple of layer kernel size requested");
        }
        weights.reshape({filter_out_dim, channel_in_dim * kernel_size});
        verticle_padding = ceil((float)filter_out_dim / arch.filter_count) * arch.filter_count - filter_out_dim;
        horizontal_padding = ceil((float)(channel_in_dim * kernel_size) / arch.channel_count) * arch.channel_count -
                             (channel_in_dim * kernel_size);
        break;
    }
    default:
        cout << "INVALID ORIENTATION" << endl;
        exit(EXIT_FAILURE);
        break;
    }

    xt::xarray<int> padded_weights =
        xt::pad(weights, {{0, verticle_padding}, {0, horizontal_padding}}, xt::pad_mode::constant, PAD);

    return padded_weights;
}

template <typename DataType>
bool validate_output(xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<DataType> arch_output)
{
    auto expected_output = evaluate_expected_output<DataType>(ifmap, weights);
    return expected_output == arch_output;
}

} // namespace LayerGeneration