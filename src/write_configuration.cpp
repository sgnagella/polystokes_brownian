#include <iostream> 
#include "Stokes.h"
#include "arrays.h"
#include <string>
#include <cstring>
#include <cstdio>

using namespace arrays;

void PolyStokes::write_configuration(){
    // Write the configuration of the particles to a file
    std::ostringstream oss;
    double& t = timeinfo.t;
    double time; 
    int i, i3;
    int& Np = pinfo.Np;
    int& ndim = consts.ndim;
    memcpy(&time, &t, sizeof(double));
    oss << output_dir << "/config_" <<  time << "_.txt"; 
    std::string filename_str = oss.str();
    const char* filename = filename_str.c_str();
    FILE* file; 
    file = fopen(filename, "w");

    if (file != NULL){
        for(i = 0; i < Np; i++){
            i3 = ndim * i;
            fprintf(file, "%.10f", x[i3]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", x[i3 + 1]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", x[i3 + 2]);
            fprintf(file, "\n");}
        fclose(file);

    }   
    else{
        std::cerr << "Error: Could not open the file for writing in write_configuration." << std::endl;
    }
    return;
}

void PolyStokes::write_quaternions(){
    // Write the configuration of the particles to a file
    std::ostringstream oss;
    double& t = timeinfo.t;
    double time; 
    int& Nc = pinfo.Nc;
    int i, i4;
    memcpy(&time, &t, sizeof(double));
    oss << output_dir << "/quats_" << time << "_.txt";
    std::string filename_str = oss.str();
    const char* filename = filename_str.c_str();
    FILE* file; 
    file = fopen(filename, "w");

    if (file != NULL){
        for(i = 0; i < Nc; i++){
            i4 = 4 * i;
            fprintf(file, "%.10f", q[i4]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", q[i4 + 1]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", q[i4 + 2]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", q[i4 + 3]);  
            fprintf(file, "\n");}
        fclose(file);
    }   
    else{
        std::cerr << "Error: Could not open the file for writing in write_quaternion." << std::endl;
    }
    return;
}

void PolyStokes::write_forces(){
    // Write the configuration of the particles to a file
    std::ostringstream oss;
    double& t = timeinfo.t;
    double time; 
    int i, i3, i3p3;
    int& Nm = pinfo.Nm;
    int& Nc = pinfo.Nc;
    int& Np = pinfo.Np;
    int& ndim = consts.ndim;
    int& n3 = consts.n3;
    memcpy(&time, &t, sizeof(double));
    oss << output_dir << "/force_" <<  time << "_.txt"; 
    std::string filename_str = oss.str();
    const char* filename = filename_str.c_str();
    FILE* file; 
    file = fopen(filename, "w");

    if (file != NULL){
        // Write monomer + colloid forces
        for(i = 0; i < Np; i++){
            i3 = ndim * i; 
            fprintf(file, "%.10f", fext[i3]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", fext[i3 + 1]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", fext[i3 + 2]);
            fprintf(file, "\n");
        }

        // Write colloid torques
        for(i = 0; i < Nc; i++){
            i3 = n3 + ndim * i;  
            fprintf(file, "%.10f", fext[i3]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", fext[i3 + 1]);
            fprintf(file, "\t");
            fprintf(file, "%.10f", fext[i3 + 2]);
            fprintf(file, "\n");
        }
        fclose(file);

    }   
    else{
        std::cerr << "Error: Could not open the file for writing in write_configuration." << std::endl;
    }
    return;
}