//
// Created by elle on 01/11/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <baylib/network/bayesian_utils.hpp>
#include "types.h"

namespace py = pybind11;

void bind_bayesian_network (py::module &m)
{
    py::class_<bayesian_network>(m, "bayesian_net")
            .def(py::init<>())
            .def("add_variable", &bayesian_network::add_variable<>)
            .def("remove_variable", &bayesian_network::remove_variable)
            .def("add_dependency", &bayesian_network::add_dependency)
            .def("remove_dependency", &bayesian_network::remove_dependency)
            .def("has_dependency", &bayesian_network::has_dependency)
            .def("number_of_variables", &bayesian_network::number_of_variables)
            .def("variable", py::overload_cast<ulong>(&bayesian_network::variable))
            .def("variable", py::overload_cast<ulong>(&bayesian_network::variable, py::const_))
            .def("is_root", &bayesian_network::is_root )
            .def("children_of", &bayesian_network::children_of)
            .def("parents_of", &bayesian_network::parents_of)
            .def("set_variable_probability", &bayesian_network::set_variable_probability)
            .def("has_variable", &bayesian_network::has_variable)
            ;
}

void bind_random_variable(py::module &m)
{
    py::class_<random_var>(m, "random_variable")
            .def(py::init<unsigned long>())
            .def("set_probability", &random_var::set_probability)
            .def("table", py::overload_cast<>(&random_var::table))
            .def("table", py::overload_cast<>(&random_var::table, py::const_))
            .def("id", &random_var::id)
            .def("number_of_states", &random_var::number_of_states)
            .def("set_as_evidence", &random_var::set_as_evidence)
            .def("clear_evidence", &random_var::clear_evidence)
            .def("is_evidence", &random_var::is_evidence)
            .def("evidence_state", &random_var::evidence_state)
            ;
}

void bind_condition (py::module &m)
{

}

void bind_condition_factory(py::module &m)
{

}

void bind_conditional_probability_table(py::module &m)
{

}


void bind_marginal_distribution (py::module &m)
{
  /*py::class_<marginal_distr>(m, "marginal_distribution")
            .def(py::init<...>())
            .def("normalize", &marginal_distr::normalize)
            .def(py::self += py::self)
            .def(py::self /= float())
            ;*/

}


void bind_bayesian_utils (py::module &m)
{
    // m.def(...)
    // m.def(...)
}