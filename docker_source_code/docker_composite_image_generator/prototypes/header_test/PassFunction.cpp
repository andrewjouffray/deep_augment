// goal is to define a class that uses a function defined in the main function and the main function is in another file
// this is really good: https://stackoverflow.com/questions/9579930/separating-class-code-into-a-header-and-cpp-file

#include "PassFunction.h"
#include <iostream> 

PassFunction::PassFunction(vector<string> params){

	PassFunction::param1 = params.at(0);
	PassFunction::param2 = params.at(1);
	PassFunction::param3 = params.at(2);
	
	}

void PassFunction::doTheThing(){
	
	PassFunction::first = randomInt(0, 5);
	PassFunction::second = randomFloat(0.0, 5.0);

	cout << "param1: " + PassFunction::param1 << endl;
	cout << "param2: " + PassFunction::param2 << endl;
	cout << "param3: " + PassFunction::param3 << endl;

	cout << PassFunction::first << endl;
	cout << PassFunction::second << endl;
	
}




