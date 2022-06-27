#include <iostream>
#include <string>
#include "PassFunction.h"
#include <random>


int main(){

	vector<string> params;

	params.push_back("param1");
	params.push_back("param2");
	params.push_back("param3");
	params.push_back("param4");
	params.push_back("param5");

	PassFunction thing = PassFunction(params);

	thing.doTheThing();



}

