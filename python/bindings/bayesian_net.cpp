//
// Created by elle on 01/11/21.
//

#include <pybind11/pybind11.h>
#include <baylib/network/bayesian_net.hpp>

namespace py = pybind11;
using namespace baylib;

typedef random_variable<double> random_var;
typedef bayesian_net<random_var> baynet_;

PYBIND11_MODULE(_pybaylib, m) {
    py::class_<baynet_>(m, "bayesian_net")
            .def(py::init<>())
            .def("add_variable", &baynet_::add_variable<>)
            .def("add_dependency", &baynet_::add_dependency)
            .def("has_dependency", &baynet_::has_dependency)
            .def("is_root", &baynet_::is_root );
}
