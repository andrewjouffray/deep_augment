#ifndef PASSFUNCTION_H
#define PASSFUNCTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include "randomFunc.h"

using namespace std;

class PassFunction{

public:

        string param1;
        string param2;
        string param3;
	int first;
	float second;

        PassFunction(vector<string> params);

        void doTheThing();



};


#endif
