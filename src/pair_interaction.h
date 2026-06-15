#ifndef PAIR_INT_H
#define PAIR_INT_H
#include <petscmat.h>
#include "params.h"
#include "data.h"
void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct);
#endif