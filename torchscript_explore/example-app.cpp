#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }


    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";

    double total_time = 0;

    for(int i = 0; i<102; i++)
    {
        auto t1 = high_resolution_clock::now();
        at::Tensor output = module.forward(inputs).toTensor();
        auto t2 = high_resolution_clock::now();

        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
        if (i > 2)
        {
            total_time += ms_double.count();
            /* code */
        }
        
    }

    std::cout << total_time / 100 << '\n';
}