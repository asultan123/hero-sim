/**
 * @file dmaTestRegisterMap.hh
 * @author Vincent Zhao
 * @brief Defines the memory layout used for the DMATester testbench.
 */

#if !defined(__DMA_TEST_ADDR_MAP_H__)
#define __DMA_TEST_ADDR_MAP_H__

// clang-format off
#define MEM_BASE_ADDR         0x1000ULL
#define MEM_SIZE              64 * 1024
#define MM2S_BASE_ADDR        0x80000000ULL
#define MM2S_REG_SIZE         0x2CU
#define S2MM_BASE_ADDR        0x80001000ULL
#define AXIDMA_CR_REG(base, ii)       ((base) + (ii) * MM2S_REG_SIZE + 0x00)
#define AXIDMA_SR_REG(base, ii)       ((base) + (ii) * MM2S_REG_SIZE + 0x04)
#define AXIDMA_ADDR_REG(base, ii)     ((base) + (ii) * MM2S_REG_SIZE + 0x18)
#define AXIDMA_ADDR_MSB_REG(base, ii) ((base) + (ii) * MM2S_REG_SIZE + 0x1C)
#define AXIDMA_LENGTH_REG(base, ii)   ((base) + (ii) * MM2S_REG_SIZE + 0x28)
#define MM2S_CR_REG(ii)       AXIDMA_CR_REG(MM2S_BASE_ADDR, ii)
#define MM2S_SR_REG(ii)       AXIDMA_SR_REG(MM2S_BASE_ADDR, ii)
#define MM2S_ADDR_REG(ii)     AXIDMA_ADDR_REG(MM2S_BASE_ADDR, ii)
#define MM2S_ADDR_MSB_REG(ii) AXIDMA_ADDR_MSB_REG(MM2S_BASE_ADDR, ii)
#define MM2S_LENGTH_REG(ii)   AXIDMA_LENGTH_REG(MM2S_BASE_ADDR, ii)
#define S2MM_CR_REG(ii)       AXIDMA_CR_REG(S2MM_BASE_ADDR, ii)
#define S2MM_SR_REG(ii)       AXIDMA_SR_REG(S2MM_BASE_ADDR, ii)
#define S2MM_ADDR_REG(ii)     AXIDMA_ADDR_REG(S2MM_BASE_ADDR, ii)
#define S2MM_ADDR_MSB_REG(ii) AXIDMA_ADDR_MSB_REG(S2MM_BASE_ADDR, ii)
#define S2MM_LENGTH_REG(ii)   AXIDMA_LENGTH_REG(S2MM_BASE_ADDR, ii)
#define AXIDMA_CR_IOC_IRQ_EN        1 << 12;
// clang-format on

#endif  // __DMA_TEST_ADDR_MAP_H__