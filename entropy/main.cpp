#include "Start.h"
#include "Examples/PythonModel.h"

using namespace PostEnt2026;

int main(int argc, char** argv)
{
    start<Examples::PythonModelParams, Examples::PythonModelData>(argc, argv);

    return 0;
}

