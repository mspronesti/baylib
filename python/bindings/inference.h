//
// Created by elle on 02/11/21.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "types.h"

namespace py = pybind11;

class py_inference_algo : public abstract_inference_algorithm
{
    typedef abstract_inference_algorithm algorithm_;
public:
    // Inherit the constructors
    using algorithm_::algorithm_;

    marginal_distr make_inference () override {
        PYBIND11_OVERLOAD_PURE(
                marginal_distr, // Return type
                algorithm_,  // Parent class
                make_inference, // Name of method
        );
    }

};

class py_parallel_algo : public _par_inference_algorithm
{
    typedef _par_inference_algorithm algorithm_;
protected:
    marginal_distr sample_step(
        ulong nsamples,
        uint seed
    ) override
    {
        PYBIND11_OVERLOAD_PURE (
              marginal_distr,
              algorithm_,
              sample_step,
              nsamples,
              seed
        );
    }
public:
    // Inherit the constructors
    using algorithm_::algorithm_;
};


template<typename Algo_>
void bind_parallel_inference_algorithm (
        py::module &m,
        const std::string & name
)
{
    typedef typename Algo_::network_type network_type;
    py::class_<
            abstract_inference_algorithm,
            py_inference_algo
            >(m, "_inference_base")
            .def(py::init <
                         network_type,
                         ulong,
                         uint
                 > ()
            )
            .def("make_inference", &abstract_inference_algorithm::make_inference)
            ;

    py::class_<
            _par_inference_algorithm,
            abstract_inference_algorithm,
            py_parallel_algo
            > (m, "_par_algo")
            .def(py::init <
                         network_type,
                         ulong,
                         uint,
                         uint
                 > ()
            )
           .def("sample_step", &_par_inference_algorithm::sample_step)
           ;

    py::class_<
            Algo_,
            _par_inference_algorithm,
            abstract_inference_algorithm
            > (m, "gibbs_sampling")
            .def(py::init <
                         network_type,
                         ulong,
                         uint,
                         uint
                 > ()
            )
            //.def("make_inference", &Algo_::make_inference)

            ;
}


