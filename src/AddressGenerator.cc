#include "AddressGenerator.hh"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

#include "Memory_Channel.hh"

namespace {
constexpr char hdr_magic_num[6] = {'N', 'N', 'P', 'R', 'O', 'G'};
}  // namespace

Descriptor_2D::Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state,
                             unsigned int _x_count, int _x_modify, unsigned int _y_count,
                             int _y_modify) {
  this->next = _next;
  this->start = _start;
  this->state = _state;
  this->x_count = _x_count;
  this->x_modify = _x_modify;
  this->y_count = _y_count;
  this->y_modify = _y_modify;
}

Descriptor_2D Descriptor_2D::default_descriptor() {
  return {0, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0};
}

bool Descriptor_2D::operator==(const Descriptor_2D& rhs) {
  return this->next == rhs.next && this->start == rhs.start && this->state == rhs.state &&
         this->x_count == rhs.x_count && this->x_modify == rhs.x_modify &&
         this->y_count == rhs.y_count && this->y_modify == rhs.y_modify;
}

template <typename DataType>
void AddressGenerator<DataType>::resetIndexingCounters() {
  x_count_remaining = descriptors[execute_index].x_count;
  y_count_remaining = descriptors[execute_index].y_count;
}

template <typename DataType>
void AddressGenerator<DataType>::loadInternalCountersFromIndex(unsigned int index) {
  current_ram_index = descriptors.at(index).start;
  x_count_remaining = descriptors.at(index).x_count;
  y_count_remaining = descriptors.at(index).y_count;
}

template <typename DataType>
void AddressGenerator<DataType>::loadProgram(const vector<Descriptor_2D>& newProgram) {
  descriptors.clear();
  copy(newProgram.begin(), newProgram.end(), std::back_inserter(descriptors));
}

template <typename DataType>
void AddressGenerator<DataType>::resetProgramMemory() {
  execute_index = 0;
  descriptors.clear();
  descriptors.push_back(Descriptor_2D::default_descriptor());
}

template <typename DataType>
Descriptor_2D AddressGenerator<DataType>::currentDescriptor() {
  return descriptors.at(execute_index);
}

template <typename DataType>
Descriptor_2D AddressGenerator<DataType>::nextDescriptor() {
  return descriptors.at(descriptors[execute_index].next);
}

template <typename DataType>
void AddressGenerator<DataType>::updateCurrentIndex() {
  if (x_count_remaining != 0) {
    x_count_remaining = x_count_remaining - 1;
  }

  if (x_count_remaining == 0) {
    if (y_count_remaining != 0) {
      current_ram_index = current_ram_index + currentDescriptor().y_modify;
      // HACK WITH CHANNEL->SET_ADDR
      channel->set_addr(current_ram_index + currentDescriptor().y_modify);
      x_count_remaining = currentDescriptor().x_count;
      y_count_remaining = y_count_remaining - 1;
    }
  } else {
    // HACK WITH CHANNEL->SET_ADDR
    current_ram_index = current_ram_index + currentDescriptor().x_modify;
    channel->set_addr(current_ram_index + currentDescriptor().x_modify);
  }
}

template <typename DataType>
bool AddressGenerator<DataType>::descriptorComplete() {
  return (x_count_remaining == 0 && y_count_remaining == 0);
}

template <typename DataType>
void AddressGenerator<DataType>::loadNextDescriptor() {
  execute_index = currentDescriptor().next;
  loadInternalCountersFromIndex(currentDescriptor().next);
  channel->set_addr(nextDescriptor().start);
}

template <typename DataType>
void AddressGenerator<DataType>::update() {
  if (control->reset()) {
    resetProgramMemory();
    loadInternalCountersFromIndex(0);
    programmed = false;
    first_cycle = false;
    channel->reset();
    std::cout << "@ " << sc_time_stamp() << " " << this->name() << ":MODULE has been reset"
              << std::endl;
  } else if (control->program()) {
    if (!programmed) {
      DataType curr_data = program_data.read();
      uint64_t curr_data_val = curr_data;
      size_t bytes_to_copy = std::min(program_buf.size() - program_buf_idx, data_length_bytes);
      if (program_buf_idx >= program_buf.size())
        throw std::runtime_error("program_buf_idx out of bounds");
      memcpy(program_buf.data() + program_buf_idx, reinterpret_cast<uint8_t*>(&curr_data_val),
             bytes_to_copy);
      program_buf_idx += bytes_to_copy;
      if (program_wait) {
        if (program_buf_idx >= program_buf.size()) {
          program_wait--;
          program_buf_idx = 0;
        }
      } else {
        // Copy this AG's program
        if (program_descriptors_left) {
          if (program_buf_idx >= sizeof(Descriptor_2D)) {
            Descriptor_2D* descriptor = reinterpret_cast<Descriptor_2D*>(program_buf.data());
            descriptors.push_back(*descriptor);
            program_descriptors_left--;
            if (program_descriptors_left == 0) {
              execute_index = 0;
              loadInternalCountersFromIndex(0);
              channel->set_addr(descriptors.at(0).start);
              first_cycle = true;
              programmed = true;
              std::cout << "@ " << sc_time_stamp() << " " << this->name()
                        << ":MODULE has been programmed" << std::endl;
            }
            program_buf_idx = 0;
          }
        } else {
          // Header parsing
          uint8_t* magic_start = static_cast<uint8_t*>(
              memchr(program_buf.data(), hdr_magic_num[0], program_buf_idx + 1));
          while (program_buf_idx >= sizeof(Program_Hdr)) {
            if (magic_start) {
              // Shift potential header start to front of buffer
              size_t new_buf_size = program_buf.data() + program_buf_idx - magic_start;
              memmove(program_buf.data(), magic_start, new_buf_size);
              program_buf_idx = new_buf_size;

              // Add remainder data if possible
              if (bytes_to_copy < data_length_bytes) {
                size_t remainder_to_add = std::min(program_buf.size() - program_buf_idx,
                                                   data_length_bytes - bytes_to_copy);
                memcpy(program_buf.data() + program_buf_idx,
                       reinterpret_cast<uint8_t*>(curr_data_val) + bytes_to_copy, remainder_to_add);
                program_buf_idx += remainder_to_add;
                bytes_to_copy += remainder_to_add;
              }

              if (program_buf_idx < sizeof(Program_Hdr))
                break;  // No longer enough data for header parsing

              if (memcmp(program_buf.data(), hdr_magic_num, sizeof(hdr_magic_num))) {
                // Magic number mismatch, find next potential header
                magic_start = static_cast<uint8_t*>(
                    memchr(program_buf.data() + 1, hdr_magic_num[0], program_buf_idx));
              } else {
                Program_Hdr* header = reinterpret_cast<Program_Hdr*>(program_buf.data());
                if (header->uid == uid) {
                  program_descriptors_left = header->num_descriptors;
                  descriptors.clear();
                } else {
                  program_wait = header->num_descriptors;
                }

                // Remove header from buffer
                memmove(program_buf.data(), program_buf.data() + program_buf_idx,
                        program_buf_idx - sizeof(Program_Hdr));
                program_buf_idx -= sizeof(Program_Hdr);

                break;
              }
            } else {
              program_buf_idx = 0;
            }
          }
        }
      }

      // Copy remaining data if needed
      if (bytes_to_copy < data_length_bytes)
        memcpy(program_buf.data() + program_buf_idx,
               reinterpret_cast<uint8_t*>(&curr_data_val) + bytes_to_copy,
               data_length_bytes - bytes_to_copy);
    }
  } else if (control->enable() && programmed) {
    // Update internal address counters, ignore for first cycle due to channel
    // enable delay
    if (!first_cycle && (currentDescriptor().state == DescriptorState::GENERATE ||
                         currentDescriptor().state == DescriptorState::WAIT)) {
      updateCurrentIndex();
      if (descriptorComplete()) {
        loadNextDescriptor();
        channel->set_enable(false);
        first_cycle = true;
      }
    } else {
      first_cycle = false;
    }

    if (!descriptorComplete()) {
      switch (currentDescriptor().state) {
        case DescriptorState::GENERATE: {
          channel->set_enable(true);
          break;
        }
        case DescriptorState::WAIT:
        case DescriptorState::SUSPENDED: {
          channel->set_enable(false);
          break;
        }
        default: {
          std::cout << "@ " << sc_time_stamp() << " " << this->name()
                    << ": Is in an invalid state! ... exitting" << std::endl;
          exit(-1);
        }
      }
    }

    // else
    // {
    //     // pre enable for next descriptors first cycle
    //     switch (nextDescriptor().state)
    //     {
    //     case DescriptorState::GENERATE:
    //     {
    //         channel->set_enable(true);
    //         break;
    //     }
    //     case DescriptorState::WAIT:
    //     case DescriptorState::SUSPENDED:
    //     {
    //         channel->set_enable(false);
    //         break;
    //     }
    //     default:
    //     {
    //         std::cout << "@ " << sc_time_stamp() << " " << this->name()
    //                 << ": Is in an invalid state! ... exitting" << std::endl;
    //         exit(-1);
    //     }
    //     }
    // }
  }

  ready = programmed && control->enable() ? currentDescriptor().state == DescriptorState::GENERATE
                                          : false;
}

// Constructor
template <typename DataType>
AddressGenerator<DataType>::AddressGenerator(sc_module_name name, GlobalControlChannel& _control,
                                             sc_trace_file* _tf, uint16_t& _uid, uint16_t& _max_uid)
    : sc_module(name),
      control("control"),
      channel("channel"),
      tf(_tf),
      execute_index("execute_index"),
      current_ram_index("current_ram_index"),
      x_count_remaining("x_count_remaining"),
      y_count_remaining("y_count_remaining") {
  control(_control);
  _clk(control->clk());
  _reset(control->reset());
  DataType test;
  static_assert(sizeof(Program_Hdr) <= sizeof(Descriptor_2D),
                "Program header must not exceed descriptor size");
  assert(test.length() % 8 == 0);
  data_length_bytes = test.length() / 8;
  assert(data_length_bytes <= program_buf.size());
  execute_index = 0;
  // sc_trace(tf, this->execute_index, (this->execute_index.name()));
  sc_trace(tf, this->execute_index, (this->execute_index.name()));
  sc_trace(tf, this->current_ram_index, (this->current_ram_index.name()));
  sc_trace(tf, this->x_count_remaining, (this->x_count_remaining.name()));
  sc_trace(tf, this->y_count_remaining, (this->y_count_remaining.name()));

  SC_METHOD(update);
  sensitive << _clk.pos();
  sensitive << _reset.pos();

  uid = _uid > _max_uid ? UINT16_MAX : _uid++;

  // connect signals
  std::cout << "ADDRESS_GENERATOR MODULE: " << name << " has been instantiated " << std::endl;
}

template <typename DataType>
AddressGeneratorCreator<DataType>::AddressGeneratorCreator(GlobalControlChannel& _control,
                                                           sc_trace_file* _tf, uint16_t& _uid,
                                                           uint16_t& _max_uid)
    : tf(_tf), control(_control), uid(_uid), max_uid(_max_uid) {}

template <typename DataType>
AddressGenerator<DataType>* AddressGeneratorCreator<DataType>::operator()(const char* name,
                                                                          size_t) {
  return new AddressGenerator<DataType>(name, control, tf, uid, max_uid);
}

template struct AddressGenerator<sc_int<32>>;
template struct AddressGeneratorCreator<sc_int<32>>;
