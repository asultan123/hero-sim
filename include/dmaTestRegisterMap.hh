#if !defined(__DMA_TEST_ADDR_MAP_H__)
#define __DMA_TEST_ADDR_MAP_H__

// clang-format off
#define MEM_BASE_ADDR         0x1000ULL
#define MEM_SIZE              64 * 1024
#define MM2S_BASE_ADDR        0x80000000ULL
#define MM2S_REG_SIZE         0x2CU
#define MM2S_CR_REG(ii)       (MM2S_BASE_ADDR + (ii) * MM2S_REG_SIZE + 0x00)
#define MM2S_SR_REG(ii)       (MM2S_BASE_ADDR + (ii) * MM2S_REG_SIZE + 0x04)
#define MM2S_ADDR_REG(ii)     (MM2S_BASE_ADDR + (ii) * MM2S_REG_SIZE + 0x18)
#define MM2S_ADDR_MSB_REG(ii) (MM2S_BASE_ADDR + (ii) * MM2S_REG_SIZE + 0x1C)
#define MM2S_LENGTH_REG(ii)   (MM2S_BASE_ADDR + (ii) * MM2S_REG_SIZE + 0x28)
#define AXIDMA_CR_IOC_IRQ_EN        1 << 12;
// clang-format on

#endif  // __DMA_TEST_ADDR_MAP_H__