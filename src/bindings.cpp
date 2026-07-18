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
        // Factory lambda so `box` can be None (no PBC) or a 3-element list [Lx,Ly,Lz].
        .def(py::init([](double dt, int samplerate, double tmax, std::string output_dir,
                         bool mm_HI, bool chain_HI, bool fene, bool record_forces, bool tether,
                         bool mono_ev, py::object box, double t, unsigned long seed, double restart) {
                 std::vector<double> box_lengths;
                 if (!box.is_none()) {
                     box_lengths = box.cast<std::vector<double>>();
                 }
                 return new PolyStokes(dt, samplerate, tmax, output_dir, mm_HI, chain_HI,
                                       fene, record_forces, tether, mono_ev, box_lengths, t,
                                       seed, restart);
             }),
             py::arg("dt"),
             py::arg("samplerate"),
             py::arg("tmax"),
             py::arg("output_dir"),
             py::arg("mm_HI")=true,
             py::arg("chain_HI")=false,
             py::arg("fene")=true,
             py::arg("record_forces")=true,
             py::arg("tether")=false,
             py::arg("mono_ev")=false,
             py::arg("box")=py::none(),
             py::arg("t")=0.0,
             py::arg("seed")=12345,
             py::arg("restart")=0.0)
        .def("initial_configuration", &PolyStokes::initial_configuration,
             py::arg("init_x0"), py::arg("init_q")=py::none(),
             "Initialize particle positions (init_x0). On restart, pass init_q (Nc x 4 flattened) "
             "to restore the saved colloid quaternions; omit it for a fresh start (identity).")
        .def("particle_info", &PolyStokes::particle_info,
             py::arg("kT"),
             py::arg("epsilon"),
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
        .def("run", &PolyStokes::run, "Run the simulation")
        .def("set_warn_neg_eig", &PolyStokes::set_warn_neg_eig, py::arg("on")=true,
             "Toggle per-event [schur] negative-eigenvalue warning prints (default on). Pass "
             "False to suppress them for speed; statistics are collected regardless and a "
             "summary is printed at the end of run().");

    // Periodic box utility: same minimum-image / wrapping logic used by the
    // simulator, exposed so analysis scripts can reuse it consistently.
    py::class_<Box>(m, "Box")
        .def(py::init<>())
        .def("configure", &Box::configure, py::arg("lengths"),
             "Configure the box: [] disables PBC, [Lx,Ly,Lz] enables an orthorhombic cell")
        .def("active", &Box::active)
        .def("minimum_image",
             [](const Box& b, double dx, double dy, double dz) {
                 b.minimum_image(dx, dy, dz);
                 return py::make_tuple(dx, dy, dz);
             },
             py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "Return the minimum-image of a separation vector")
        .def("wrap",
             [](const Box& b, double x, double y, double z) {
                 b.wrap(x, y, z);
                 return py::make_tuple(x, y, z);
             },
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Return a position wrapped into the primary cell centered at the origin");
}
