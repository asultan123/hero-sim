#include "Connector.cpp"
#include <systemc.h>

// #define DEBUG

using std::cout;
using std::endl;

template <typename DataType>
struct DUMMY_SUB_MODULE_OUT : public sc_module
{
    sc_out<DataType> dummy_sub_module_out;

    DUMMY_SUB_MODULE_OUT(
        sc_module_name name) : sc_module(name), dummy_sub_module_out((string(name) + "dummy_sub_module_out").c_str())
    {
        cout << "DUMMY_SUB_MODULE_OUT MODULE: " << name << " has been instantiated " << endl;
    }
};

template <typename DataType>
struct DUMMY_SUB_MODULE_IN : public sc_module
{
    sc_in<DataType> dummy_sub_module_in;

    DUMMY_SUB_MODULE_IN(
        sc_module_name name) : sc_module(name), dummy_sub_module_in((string(name) + "dummy_sub_module_in").c_str())
    {
        cout << "DUMMY_SUB_MODULE_IN MODULE: " << name << " has been instantiated " << endl;
    }
};

template <typename DataType>
struct DUMMY_MODULE : public sc_module
{
    Connector connector;
    DUMMY_SUB_MODULE_OUT<DataType> sub_out{"sub_out"};
    DUMMY_SUB_MODULE_IN<DataType> sub_in{"sub_in"};
    // Constructor
    DUMMY_MODULE(
        sc_module_name name, sc_trace_file* tf) : sc_module(name), connector((string(name) + "_connector").c_str(), tf)
    {
        connector.add((string(name) + "_connection").c_str(), sub_out.dummy_sub_module_out, sub_in.dummy_sub_module_in);
        cout << "DUMMY_MODULE MODULE: " << name << " has been instantiated " << endl;
    }
};

template <typename DataType>
struct Connector_TB : public sc_module
{
    const unsigned int width{8};

    sc_trace_file* tf{sc_create_vcd_trace_file("ProgTrace")};

    // heirarchy exploration
    DUMMY_MODULE<DataType> module1{"dummy_module_1", tf};
    DUMMY_MODULE<DataType> module2{"dummy_module_2", tf};

    Connector connector{"testbench_connector", tf};

    sc_out<DataType> one_to_one_port_out{"one_to_one_port_out"};
    sc_in<DataType> one_to_one_port_in{"one_to_one_port_in"};
    sc_in<DataType> one_to_one_port_in_reuse{"one_to_one_port_in_reuse"};

    // error edge case testing, need to make this form of testing more formal
    // using exceptions or something
    
    sc_signal<DataType> one_to_one_port_out_other_binder{"one_to_one_port_out_other_binder"};
    sc_out<DataType> one_to_one_port_out_other{"one_to_one_port_out_other"};
    sc_in<DataType> one_to_one_port_in_reuse_other{"one_to_one_port_in_reuse_other"};
    
    // sc_out<unsigned int> one_to_one_port_out_error{"one_to_one_port_out_error"};
    // sc_in<unsigned int> one_to_one_port_in_error{"one_to_one_port_in_error"};

    sc_out<DataType> one_to_many_port_out{"one_to_many_port_out"};
    sc_vector<sc_in<DataType>> one_to_many_ports_in{"one_to_many_ports_in", width};
    sc_vector<sc_in<DataType>> one_to_many_ports_in_reuse{"one_to_many_ports_in_reuse", width};

    sc_vector<sc_out<DataType>> many_to_many_ports_out{"many_to_many_ports_out", width};
    sc_vector<sc_in<DataType>> many_to_many_ports_in{"many_to_many_ports_in", width};
    sc_vector<sc_in<DataType>> many_to_many_ports_in_reuse{"many_to_many_ports_in_reuse", width};

    Connector_TB(sc_module_name name) : sc_module(name)
    {
        connector.add("one_to_one", one_to_one_port_out, one_to_one_port_in);
        connector.add("one_to_one", one_to_one_port_out, one_to_one_port_in_reuse);

        one_to_one_port_out_other.bind(one_to_one_port_out_other_binder);
        connector.add("one_to_one", one_to_one_port_out_other, one_to_one_port_in_reuse_other);
        
        // illegal connection reuse to wrong type caught by System C elaboration 
        // connector.add("one_to_one", one_to_one_port_out_error, one_to_one_port_in_error);

        connector.add("one_to_many", one_to_many_port_out, one_to_many_ports_in);
        connector.add("one_to_many", one_to_many_port_out, one_to_many_ports_in_reuse);

        connector.add("many_to_many", many_to_many_ports_out, many_to_many_ports_in);
        connector.add("many_to_many", many_to_many_ports_out, many_to_many_ports_in_reuse);

        cout << "Instantiated Connector TB with name " << this->name() << endl;
        tf->set_time_unit(1, SC_PS);
    }

    bool validate_connector_write()
    {
        cout << "Validating validate_connector_write" << endl;

        cout << connector.connection_tracker["one_to_one"]->name() << endl;

        sc_start(1, SC_NS);

        cout << "Writing to output ports" << endl;
        one_to_one_port_out.write(DataType(1));
        one_to_many_port_out.write(DataType(1));
        for (auto& port : many_to_many_ports_out)
        {
            port.write(DataType(1));
        }
        sc_start(1, SC_NS);

        cout << "Validating writes from input ports" << endl;

        //one to one
        if (!(one_to_one_port_in.read() == DataType(1)))
        {
            cout << "one_to_many_ports_in.read() == DataType(1) FAILED!" << endl;
            return -1;
        }
        if (!(one_to_one_port_in_reuse.read() == DataType(1)))
        {
            cout << "one_to_many_ports_in_reuse.read() == DataType(1) FAILED!" << endl;
            return -1;
        }

        // one to many
        for (auto& port : one_to_many_ports_in)
        {
            if (!(port.read() == DataType(1)))
            {
                cout << "port.read() == DataType(1) FAILED!" << endl;
                return -1;
            }
        }
        for (auto& port : one_to_many_ports_in_reuse)
        {
            if (!(port.read() == DataType(1)))
            {
                cout << "port.read() == DataType(1) FAILED!" << endl;
                return -1;
            }
        }

        // many to many
        for (auto& port : many_to_many_ports_in)
        {
            if (!(port.read() == DataType(1)))
            {
                cout << "port.read() == DataType(1) FAILED!" << endl;
                return -1;
            }
        }
        for (auto& port : many_to_many_ports_in_reuse)
        {
            if (!(port.read() == DataType(1)))
            {
                cout << "port.read() == DataType(1) FAILED!" << endl;
                return -1;
            }
        }

        cout << "validate_connector_write SUCCESS" << endl;
        return true;
    }

    int run_tb()
    {
        if (!(validate_connector_write()))
        {
            cout << "validate_connector_write() FAILED!" << endl;
            return -1;
        }
        cout << "TEST BENCH SUCCESS " << endl;
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
    ~Connector_TB()
    {
        sc_close_vcd_trace_file(tf);
    }
};
int sc_main(int argc, char* argv[])
{
    Connector_TB<sc_int<32>> tb("Connector_tb");
    return tb.run_tb();
}