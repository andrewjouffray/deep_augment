#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include "../include/Dataset.h"

namespace fs = std::filesystem;
using namespace std;

int main(int argc, char** argv){

	if (argc == 1){
	
		cout << "Error: No argument entered" << endl;
		cout << "options: " << endl;
		cout << "	-init: initialize a new dataset and enter the augmentation options" << endl;
		cout << "	-run: runs the data augmentation algorithm" << endl;
		return 0;
	}

	for (int i = 1; i < argc; ++i){

        	string command  = argv[i];

		// check the command entered by the user
		if (command.compare("-run") == 0){

			string path;

			// if user specified a path
			if (argc > 2){
				path = argv[i+1];

				// add a / if there are none at the end.
				if (path.substr(path.length() -1).compare("/") != 0){
			
					path = path + "/";
				}
		
			}

			// default to local path
			else{
				path = "./";
			
			}

			// run the dataset
			try{
        				
				Dataset dataset = Dataset(path);

			}catch(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >& e){
			
				cout << e << endl;		
			}

			return 0;
		
		}else if(command.compare("-init") == 0){
		
			cout << "Not implemented yet" << endl;
		}else{
		
			cout << "Error: wrong command" << endl;
		}
	}

        return 0;
}

