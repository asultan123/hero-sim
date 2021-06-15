/**
 * @file MultiProgramSender.hh
 * @author Vincent Zhao
 * @brief Serializes multiple address generator programs and clocks the bytestream out.
 */

#if !defined(__MULTIPROGRAMSENDER_H__)
#define __MULTIPROGRAMSENDER_H__

#include <cstddef>
#include <cstdint>
#include <systemc>
#include <vector>

using namespace sc_core;

// Forward declarations
struct Descriptor_2D;
struct GlobalControlChannel_IF;

template <typename DataType>
class MultiProgramSender : public sc_module {
  SC_HAS_PROCESS(MultiProgramSender);

 public:
  MultiProgramSender(GlobalControlChannel_IF& control, sc_trace_file* tf, uint16_t startUid = 0,
                     sc_module_name moduleName = "multi-program-sender");
  void addProgram(std::vector<Descriptor_2D>& program);

  sc_out<DataType> output;

 private:
  void sendPrograms();
  template <typename T>
  void addToStream(T& item);

  std::vector<uint8_t> bytestream;
  sc_in<bool> clk;
  uint16_t currentUid;
  size_t dataLengthBytes;
};

#endif  // __MULTIPROGRAMSENDER_H__
