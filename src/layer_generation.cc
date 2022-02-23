#include "layer_generation.hh"

namespace LayerGeneration
{

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

    bool validate_output_1x1(xt::xarray<int> ifmap, xt::xarray<int> weights, xt::xarray<int> arch_output)
    {
        // weights.shape() = F*C*K*K
        assert(weights.shape().size() == 4);
        // cout << xt::adapt(weights.shape()) << endl;
        // ifmap.shape() = C*H*W
        assert(ifmap.shape().size() == 3);
        // cout << xt::adapt(ifmap.shape()) << endl;
        // cout << ifmap << endl;
        // symmetric kernel
        assert(weights.shape(3) == weights.shape(2));

        // ifmap channel = weights channel in
        assert(ifmap.shape(0) == weights.shape(1));
        int ifmap_w = ifmap.shape(2);
        int ifmap_h = ifmap.shape(1);

        int kernel = weights.shape(3);
        assert(ifmap_w >= kernel);
        assert(ifmap_h >= kernel);

        int ofmap_w = ifmap_w - (kernel - 1);
        int ofmap_h = ifmap_h - (kernel - 1);
        int ofmap_c = weights.shape(0);

        xt::xarray<int> ofmap = xt::arange(ofmap_c * ofmap_w * ofmap_h).reshape({ofmap_c, ofmap_h, ofmap_w});

        // conv2d stride 1
        for (auto f = 0; f < ofmap_c; f++)
        {
            auto weight_tensor_view = xt::view(weights, f, xt::all(), xt::all(), xt::all());
            xt::xarray<int> flatten_weight(xt::flatten(weight_tensor_view));
            for (auto h = 0; h < ofmap_h; h++)
            {
                for (auto w = 0; w < ofmap_w; w++)
                {
                    auto ifmap_tensor_view = xt::view(ifmap, xt::all(), xt::range(h, h + kernel), xt::range(w, w + kernel));
                    xt::xarray<int> flattened_ifmap(xt::flatten(ifmap_tensor_view));
                    auto val = xt::linalg::dot(flattened_ifmap, flatten_weight);
                    ofmap(f, h, w) = val(0);
                }
            }
        }

        return ofmap == arch_output;
    }

    template xt::xarray<int> generate_weights<sc_int<32>>(int filter_out_dim, int channel_in_dim, int kernel);
    template xt::xarray<int> generate_ifmap<sc_int<32>>(Arch<sc_int<32>> &arch, int channel_in, int ifmap_h, int ifmap_w);
    template xt::xarray<int> pad_weights<sc_int<32>>(Arch<sc_int<32>> &arch, xt::xarray<int> weights, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation);

}