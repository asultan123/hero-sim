#include "sock2sig.hh"

#include <algorithm>

template <unsigned int BUSWIDTH>
Sock2Sig<BUSWIDTH>::Sock2Sig(int readyDelay, sc_module_name moduleName)
    : sc_module(moduleName), bitOffset(0), byteOffset(0), readyDelay(readyDelay)
{
    if (BUSWIDTH % 8 != 0)
        throw std::runtime_error("Adapter does not currently support non-byte aligned widths");
    inputSock.register_b_transport(this, &Sock2Sig<BUSWIDTH>::inputSock_b_transport);
    SC_THREAD(updateOutput);
}

template <unsigned int BUSWIDTH>
void Sock2Sig<BUSWIDTH>::inputSock_b_transport(tlm::tlm_generic_payload &trans, sc_time &delay)
{
    // Should only propagate writes
    if (trans.get_command() != tlm::tlm_command::TLM_WRITE_COMMAND || trans.get_data_length() <= 0)
    {
        std::cout << "Transaction is invalid, discarding..." << std::endl;
        return;
    }

    if (currentData)
        throw std::runtime_error("Previous transaction data still present");

    // Cache received data for later (may be wider than the bus)
    currentData = std::make_unique<std::vector<uint8_t>>(trans.get_data_length());
    memcpy(currentData->data(), trans.get_data_ptr(), currentData->size());

    dataAvailable.notify();
    wait(transComplete);
    trans.set_response_status(tlm::TLM_OK_RESPONSE);
}

template <unsigned int BUSWIDTH> void Sock2Sig<BUSWIDTH>::updateOutput()
{
    while (true)
    {
        // Deassert data ready signal
        dataReady = false;

        if (!currentData)
        {
            // No new data is available to update output, wait until more is sent
            wait(dataAvailable);
            if (!currentData)
                throw std::runtime_error("No transaction data available");
        }

        // Copy the smaller of either the closest number of bytes that fits bus
        // width or bytes left in packet
        size_t bytesToCopy = std::min(static_cast<size_t>(BUSWIDTH % 8 ? (BUSWIDTH + 8) / 8 : BUSWIDTH / 8),
                                      currentData->size() - byteOffset);

        uint64_t value = 0;

        memcpy(&value, &(*currentData)[byteOffset], bytesToCopy);

        // TODO: Non-byte aligned widths, revisit when needed
        // // Trim bits already read
        // value <<= bitOffset;

        // // Trim trailing bits
        // value >>= (64 - BUSWIDTH);
        // value <<= (64 - BUSWIDTH);
        // bitOffset = (bitOffset + BUSWIDTH) % 8;

        outputSig = sc_int<BUSWIDTH>(value);

        byteOffset += (bitOffset + BUSWIDTH) / 8;

        // currentData consumed, reset state and finish transaction
        if (byteOffset >= currentData->size())
        {
            currentData.reset();
            byteOffset = 0;
            bitOffset = 0;
            transComplete.notify();
        }

        // Wait readyDelay cycles before asserting data is ready for reading
        wait(readyDelay, SC_NS);

        dataReady = true;

        // Wait for peripheral to read and acknowledge data
        wait(assertRead.posedge_event());
    }
}

template class Sock2Sig<8>;
template class Sock2Sig<32>;
template class Sock2Sig<64>;