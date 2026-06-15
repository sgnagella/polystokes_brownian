#ifndef SADDLE_H
#define SADDLE_H
#include "globals.h"
// Routines for saddle point method
void saddle(); 
PetscErrorCode PreconditionerApply(PC pc, Vec x, Vec y);
#endif