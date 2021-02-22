#if !defined(__STRINGPRODUCER_H__)
#define __STRINGPRODUCER_H__

#include <tlm_utils/simple_initiator_socket.h>

#include <systemc>

using namespace sc_core;
using namespace sc_dt;

class StringProducer : public sc_module {
  SC_HAS_PROCESS(StringProducer);

 public:
  StringProducer(sc_module_name moduleName = "string-producer");

  tlm_utils::simple_initiator_socket<StringProducer, 8> outputSock;

 private:
  void sendString();

  tlm::tlm_generic_payload trans;
};

#endif  // __STRINGPRODUCER_H__