#include <systemc>

#include "demo-dma.h"
#include "iconnect.h"
#include "xilinx-axidma.h"

int sc_main(int argc, char *argv[])
{
    axidma dma("axidma");
    axidma_mm2s mm2s("axidma-mm2s");
    axidma_s2mm s2mm("axidma-s2mm");
    iconnect<5, 10> bus("bus");
    demodma demoDma("demo-dma");

    return 0;
}