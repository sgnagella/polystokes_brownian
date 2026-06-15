#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Stokes.h"
#include <iostream>

namespace py = pybind11;

// PYBIND11_MODULE(PolyStokes, m) {
//      m.doc() = "Polymer Stokesian Dynamics simulation module";
 
//      py::class_<PolyStokes>(m, "PolyStokes")
//          .def(py::init([](double dt, int samplerate, double tmax, const std::string& output_dir, bool mm_HI) {
//              return PolyStokes(dt, samplerate, tmax, output_dir, mm_HI);
//          }),
//          py::arg("dt"),
//          py::arg("samplerate"),
//          py::arg("tmax"),
//          py::arg("output_dir"),
//          py::arg("mm_HI") = true
//          )
//          .def("initial_configuration", &PolyStokes::initial_configuration, 
//               py::arg("init_x0"), "Initialize the configuration of the particles")
//          .def("particle_info", &PolyStokes::particle_info, 
//               py::arg("Np"), 
//               py::arg("Nc"), 
//               py::arg("Nm"), 
//               py::arg("Npoly"), 
//               py::arg("Nmono_per_chain"), 
//               py::arg("beta"), 
//               py::arg("kbond"), 
//               py::arg("r0"), 
//               py::arg("Lmax"), 
//               py::arg("tau"), "Set the parameters for the particles")
//          .def("trap_info", &PolyStokes::trap_info,
//               py::arg("ktrap"), 
//               py::arg("tstart"), 
//               py::arg("trun"), "Set the parameters for the traps")
//          .def("run", &PolyStokes::run, "Run the simulation");
//  } 

PYBIND11_MODULE(PolyStokes, m) {
    m.doc() = "Polymer Stokesian Dynamics simulation module";
    py::class_<PolyStokes>(m,"PolyStokes")
        .def(py::init<double, int, double, std::string, bool, bool, bool, bool, double>(),
             py::arg("dt"), 
             py::arg("samplerate"),
             py::arg("tmax"),
             py::arg("output_dir"),
             py::arg("mm_HI")=true,
             py::arg("chain_HI")=false,
             py::arg("fene")=true,
             py::arg("record_forces")=true,
             py::arg("t")=0.0)
        .def("initial_configuration", &PolyStokes::initial_configuration, 
             py::arg("init_x0"), "Initialize the configuration of the particles")
        .def("particle_info", &PolyStokes::particle_info, 
             py::arg("Np"), 
             py::arg("Nc"), 
             py::arg("Nm"), 
             py::arg("Npoly"), 
             py::arg("Nmono_per_chain"), 
             py::arg("beta"), 
             py::arg("kbond"), 
             py::arg("r0"), 
             py::arg("Lmax"), 
             py::arg("tau"), "Set the parameters for the particles")
        .def("trap_info", &PolyStokes::trap_info,
             py::arg("ktrap"), 
             py::arg("tstart"), 
             py::arg("trun"), 
             py::arg("weaken_trap")=-1, "Set the parameters for the traps")
        .def("run", &PolyStokes::run, "Run the simulation");
}
