#include "stringProducer.hh"

StringProducer::StringProducer(sc_module_name moduleName)
    : sc_module(moduleName) {
  SC_THREAD(sendString);
}

void StringProducer::sendString() {
  std::string testData("Hello World!");
  sc_time transportTime = SC_ZERO_TIME;
  trans.reset();
  trans.set_write();
  trans.set_data_ptr(
      reinterpret_cast<unsigned char*>(const_cast<char*>(testData.c_str())));
  trans.set_data_length(testData.size());
  outputSock->b_transport(trans, transportTime);
}