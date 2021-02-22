#include "Connector.hh"

template <typename DataType>
Connection<DataType>::Connection(sc_module_name name, sc_trace_file* _tf, sc_out<DataType>& out, sc_in<DataType>& in) : sc_module(name),
                                                                                                    signals("signal", 1)
{
    out.bind(signals[0]);
    in.bind(signals[0]);
    sc_trace(_tf, signals[0], signals[0].name());
    output_port_name = out.name();
    std::cout << "CONNECTION " << name << " instantiated and resolved to one-to-one connection" << std::endl;
}

template <typename DataType>

Connection<DataType>::Connection(sc_module_name name, sc_trace_file* _tf, sc_out<DataType>& out, sc_vector<sc_in<DataType>>& in) : sc_module(name),
                                                                                                                signals("signal", 1)
{
    out.bind(signals[0]);
    for (unsigned int idx = 0; idx < in.size(); idx++)
    {
        in[idx].bind(signals[0]);
    }
    output_port_name = out.name();
    sc_trace(_tf, signals[0], signals[0].name());
    std::cout << "CONNECTION  " << name << " instantiated and resolved to one-to-many connection" << std::endl;
}

template <typename DataType>
Connection<DataType>::Connection(sc_module_name name, sc_trace_file* _tf, sc_vector<sc_out<DataType>>& out, sc_vector<sc_in<DataType>>& in) : sc_module(name),
                                                                                                                        signals("signal", in.size())
{
    assert(out.size() == in.size());
    for (unsigned int idx = 0; idx < in.size(); idx++)
    {
        in[idx].bind(signals[idx]);
        out[idx].bind(signals[idx]);
        sc_trace(_tf, signals[idx], signals[idx].name());
    }
    output_port_name = out.name();
    std::cout << "CONNECTION " << name << " instantiated and resolved to many-to-many connection" << std::endl;
}

template <typename DataType>
void Connection<DataType>::one_to_one_rebind_to(sc_in<DataType>& target)
{
    target.bind(signals[0]);
}

template <typename DataType>
void Connection<DataType>::one_to_many_rebind_to(sc_vector<sc_in<DataType>>& targets)
{
    for (auto& target : targets)
    {
        target.bind(signals[0]);
    }
}

template <typename DataType>
void Connection<DataType>::many_to_many_rebind_to(sc_vector<sc_in<DataType>>& targets)
{
    assert(signals.size() == targets.size());
    for (unsigned int idx = 0; idx < targets.size(); idx++)
    {
        targets[idx].bind(signals[idx]);
    }
}

Connector::Connector(sc_module_name name, sc_trace_file* _tf) : sc_module(name), tf(_tf)
{
    std::cout << "CONNECTOR  " << name << " instantiated " << std::endl;
}


template <typename DataType>
void Connector::add(sc_module_name name, sc_out<DataType>& out, sc_in<DataType>& in)
{
    auto old_connection = connection_tracker.find((string)name);
    if (old_connection != connection_tracker.end())
    {
        auto downcasted_connection = dynamic_cast<Connection<DataType>*>(old_connection->second.get());
        assert(downcasted_connection);
        if (out.name() != downcasted_connection->output_port_name)
        {
            string error_msg = "Rebinding previously instantated connection currently bound to output: ";
            error_msg += downcasted_connection->output_port_name;
            error_msg += "\n";
            error_msg += "Connection is not bound to the output port passed named : ";
            error_msg += (string)out.name();
            SC_REPORT_WARNING("", error_msg.c_str());
        }
        downcasted_connection->one_to_one_rebind_to(in);
    }
    else
    {
        connection_tracker[(string)name] = unique_ptr<Connection<DataType>>(new Connection<DataType>(name, tf, out, in));
    }
}

template <typename DataType>
void Connector::add(sc_module_name name, sc_out<DataType>& out, sc_vector<sc_in<DataType>>& in)
{
    auto old_connection = connection_tracker.find((string)name);
    if (old_connection != connection_tracker.end())
    {
        auto downcasted_connection = dynamic_cast<Connection<DataType>*>(old_connection->second.get());
        assert(downcasted_connection);
        if (out.name() != downcasted_connection->output_port_name)
        {
            string error_msg = "Rebinding previously instantated connection currently bound to output: ";
            error_msg += downcasted_connection->output_port_name;
            error_msg += "\n";
            error_msg += "Connection is not bound to the output port passed with name : ";
            error_msg += (string)out.name();
            SC_REPORT_WARNING("", error_msg.c_str());
        }
        downcasted_connection->one_to_many_rebind_to(in);
    }
    else
    {
        connection_tracker[(string)name] = unique_ptr<Connection<DataType>>(new Connection<DataType>(name, tf, out, in));
    }
}


template <typename DataType>
void Connector::add(sc_module_name name, sc_vector<sc_out<DataType>>& out, sc_vector<sc_in<DataType>>& in)
{
    auto old_connection = connection_tracker.find((string)name);
    if (old_connection != connection_tracker.end())
    {
        auto downcasted_connection = dynamic_cast<Connection<DataType>*>(old_connection->second.get());
        assert(downcasted_connection);
        if (out.name() != downcasted_connection->output_port_name)
        {
            string error_msg = "Rebinding previously instantated connection currently bound to output: ";
            error_msg += downcasted_connection->output_port_name;
            error_msg += "\n";
            error_msg += "Connection is not bound to the output port passed named : ";
            error_msg += (string)out.name();
            SC_REPORT_WARNING("", error_msg.c_str());
        }
        downcasted_connection->many_to_many_rebind_to(in);
    }
    else
    {
        connection_tracker[(string)name] = unique_ptr<Connection<DataType>>(new Connection<DataType>(name, tf, out, in));
    }
}

template struct Connection<sc_int<32>>;
