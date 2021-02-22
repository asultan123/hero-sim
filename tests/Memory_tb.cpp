#include <systemc.h>
#include "Memory.cpp"

// #define DEBUG

using std::cout;
using std::endl;

template <typename DataType>
struct Memory_TB : public sc_module
{
	const unsigned int ram_length = 256;
	const unsigned int ram_width = 4;
	const unsigned int read_channel_count = 1;
	const unsigned int write_channel_count = 1;
	const unsigned int channel_count = read_channel_count + write_channel_count;

	sc_trace_file *tf;

	GlobalControlChannel control;

	MemoryChannel<DataType> rchannel;
	MemoryChannel<DataType> wchannel;
	sc_vector<sc_signal<DataType>> payload;

	Memory<DataType> mem;

	Memory_TB(sc_module_name name) : sc_module(name),
									 tf(sc_create_vcd_trace_file("Prog_trace")),
									 control("global_control_channel", sc_time(1, SC_NS), tf),
									 rchannel("read_channel", ram_width, tf),
									 wchannel("write_channel", ram_width, tf),
									 payload("payload", ram_width),
									 mem("sram",
										 control,
										 channel_count,
										 ram_length,
										 ram_width,
										 tf)
	{
		tf->set_time_unit(1, SC_PS);
		mem.channels[0](rchannel);
		mem.channels[1](wchannel);
		cout << "Instantiated Memory TB with name " << this->name() << endl;
	}

	bool validate_reset()
	{
		control.set_reset(true);
		control.set_enable(false);

		sc_start(1, SC_NS);

		control.set_reset(false);

		for (auto &row : mem.ram)
		{
			for (auto &col : row)
			{
				if (col != DataType(0))
				{
					return false;
				}
			}
		}

		return true;
	}

	bool validate_write()
	{
		control.set_enable(true);
		wchannel.set_enable(true);
		wchannel.set_mode(MemoryChannelMode::WRITE);

		cout << "writing test data to memory " << endl;
		unsigned int val = 1;
		for (unsigned int i = 0; i < ram_length; i++)
		{
			#ifdef DEBUG
			cout << "Generating Payload " << endl;
			#endif
			for (unsigned int j = 0; j < ram_width; j++)
			{
				payload[j] = val++;
			}
			sc_start(1, SC_NS);

			#ifdef DEBUG
			cout << "Writing Payload to wchannel: " << endl;
			for (auto &data : payload)
			{
				cout << data << " ";
			}
			cout << endl;
			#endif

			wchannel.channel_write_data(payload);
			wchannel.set_addr(i);

			#ifdef DEBUG
			cout << "Writing wchannel data to memory: " << endl;
			#endif

			sc_start(1, SC_NS);
		}

		cout << "validating ... ";
		unsigned int expected_data = 1;
		for (const auto &row : mem.ram)
		{
			for (const auto &col : row)
			{
				if (col != DataType(expected_data++))
				{
					return false;
				}
			}
		}
		cout << " success!" << endl;
		val = 1;

		cout << "writing to specific data elements ... " << endl;
		for (unsigned int i = 0; i < ram_length; i++)
		{
			wchannel.channel_write_data_element(val, 0);
			wchannel.channel_write_data_element(0, 1);
			wchannel.channel_write_data_element(0, 2);
			wchannel.channel_write_data_element(0, 3);
			wchannel.set_addr(i);

			sc_start(1, SC_NS);
			val += 4;
		}

		cout << "validating ... ";;
		expected_data = 1;
		for (const auto &row : mem.ram)
		{
			if (row[0] != DataType(expected_data))
			{
				return false;
			}
			expected_data += 4;
		}
		cout << " success!" << endl;

		return true;
	}

	bool validate_read()
	{
		control.set_reset(true);
		sc_start(1, SC_NS);
		control.set_reset(false);

		cout << "resetting memory " << endl;
		if (!validate_reset())
		{
			cout << "Reset Failed" << endl;
			return -1;
		}

		control.set_enable(true);
		wchannel.set_enable(true);
		wchannel.set_mode(MemoryChannelMode::WRITE);
		rchannel.set_enable(true);
		rchannel.set_mode(MemoryChannelMode::READ);
		rchannel.set_addr(0);

		cout << "writing payload to memory " <<endl;
		unsigned int val = 1;
		for (unsigned int i = 0; i < ram_length; i++)
		{
			for (unsigned int j = 0; j < ram_width; j++)
			{
				payload[j] = val++;
			}
			sc_start(1, SC_NS);

			wchannel.channel_write_data(payload);
			wchannel.set_addr(i);

			sc_start(1, SC_NS);
		}

		cout << "validating payload read from memory " << endl;
		wchannel.set_enable(false);
		unsigned int expected_data = 1;
		for (unsigned int i = 0; i < ram_length; i++)
		{
			rchannel.set_addr(i);
			sc_start(1, SC_NS);

			const sc_vector<sc_signal<DataType>> &payload = rchannel.channel_read_data();
			for (unsigned int j = 0; j < ram_width; j++)
			{
				if (payload[j] != DataType(expected_data++))
				{
					return false;
				}
			}
		}

		return true;
	}

	int run_tb()
	{
		cout << "Validating Reset" << endl;
		if (!validate_reset())
		{
			cout << "Reset Failed" << endl;
			return -1;
		}
		cout << "Reset Success" << endl;
		
		cout << "Validating Write" << endl;
		if (!validate_write())
		{
			cout << "Write Failed" << endl;
			return -1;
		}
		cout << "Write Success" << endl;

		cout << "Validating Read" << endl;
		if (!validate_read())
		{
			cout << "Read Failed" << endl;
			return -1;
		}
		cout << "Read Success" << endl;

		cout << "TEST BENCH SUCCESS " << endl
			 << endl;

		cout << "       aOOOOOOOOOOa" << endl;
		cout << "     aOOOOOOOOOOOOOOa" << endl;
		cout << "   aOO    OOOOOO    OOa" << endl;
		cout << "  aOOOOOOOOOOOOOOOOOOOa" << endl;
		cout << " aOOOOO   OOOOOO   OOOOOa" << endl;
		cout << "aOOOOO     OOOO     OOOOOa" << endl;
		cout << "aOOOOOOOOOOOOOOOOOOOOOOOOa" << endl;
		cout << "aOOOOOOOOOOOOOOOOOOOOOOOOa" << endl;
		cout << "aOOOOO   OOOOOOOO   OOOOOa" << endl;
		cout << " aOOOOO    OOOO    OOOOOa" << endl;
		cout << "  aOOOOO          OOOOOa" << endl;
		cout << "   aOOOOOOOOOOOOOOOOOOa" << endl;
		cout << "     aOOOOOOOOOOOOOOa" << endl;
		cout << "       aOOOOOOOOOOa" << endl;
	
		return 0;

	}

	~Memory_TB()
	{
		sc_close_vcd_trace_file(tf);
	}
};

int sc_main(int argc, char *argv[])
{
	Memory_TB<sc_int<32>> tb("memory_tb");
	return tb.run_tb();
}
