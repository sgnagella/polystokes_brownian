set(MODULE_NAME PolyStokes)

cmake_minimum_required(VERSION 3.15)
project(${MODULE_NAME} LANGUAGES CXX)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
# set(MPI_GUESS_LIBRARY_NAME MPICH2) # or MSMPI
# find_package(MPI REQUIRED)

# Manually set PETSC_DIR and PETSC_ARCH
set(PETSC_DIR "/home/groups/stakator/snagella/petsc")
set(PETSC_ARCH "arch-linux-c-opt")

# Find PETSc via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET petsc)

# Include the PETSc CMake configuration files
# find_package(PETSc REQUIRED HINTS ${PETSC_DIR})

# Find Pybind11
find_package(pybind11 REQUIRED)

# Include the necessary directories for PETSc and MPI
# target_include_directories(${MODULE_NAME} PRIVATE
#     ${PETSC_DIR}/${PETSC_ARCH}/include  # PETSc include path
#     ${MPI_CXX_INCLUDE_DIRS}             # MPI include path
#     ${PROJECT_SOURCE_DIR}/include       # Project's own include directory
# )

# Include directories
include_directories(
    ${PETSC_INCLUDE_DIRS}
    # ${PETSC_DIR}/${PETSC_ARCH}/include
    # ${PETSC_DIR}/include
    # ${MPI_CXX_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# # Add PETSc library path
# link_directories(${PETSC_DIR}/${PETSC_ARCH}/lib)

# # Specify Petsc libraries 
# set(PETSC_LIBRARIES
#     petsc
#     # Add any other libraries you may need (e.g., PetscLibs like lapack, blas, etc.)
# )

# add_library(PETSC_LIBRARIES OBJECT ${PETSC_DIR}/${PETSC_ARCH}/lib)
message(STATUS "PETSC_DIR: ${PETSC_DIR}")
message(STATUS "PETSC_ARCH: ${PETSC_ARCH}")
message(STATUS "PETSC Include Directory: ${PETSC_DIR}/${PETSC_ARCH}/include")

# Source files
set(SOURCES
    src/Stokes.cpp
    src/run.cpp
    src/init.cpp
    src/arrays.cpp
    src/check_dist.cpp
    src/multi_arrays_initialize.cpp
    src/mob.cpp
    src/rhs.cpp
    src/vel.cpp
    src/traj.cpp
    src/write_configuration.cpp
    src/quat.cpp
    src/pair_interaction.cpp
    src/cleanup.cpp
    src/bindings.cpp
)


# Define the Python module
pybind11_add_module(${MODULE_NAME} ${SOURCES})

# Link PETSc and MPI
target_link_libraries(PolyStokes
    PRIVATE
    PkgConfig::PETSc
    # ${PETSC_LIBRARIES}
    # # ${PETSC_DIR}/lib
    # # ${PETSC_DIR}/${PETSC_ARCH}/lib
    # # ${PETSC_DIR}/${PETSC_ARCH}/lib
    # MPI::MPI_CXX
)

# link_directories(${PETSC_DIR}/${PETSC_ARCH}/lib)

# Apply PETSc compile flags
target_compile_options(${MODULE_NAME} PRIVATE ${PETSC_CFLAGS_OTHER})
target_compile_definitions(${MODULE_NAME} PRIVATE ${PETSC_DEFINES})

# Ensure proper installation of the compiled Python module (.so)
include(GNUInstallDirs)

install(TARGETS ${MODULE_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
