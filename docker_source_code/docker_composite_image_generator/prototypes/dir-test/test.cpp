#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include "randomFunc.h"
#include <sys/stat.h>

// https://en.cppreference.com/w/cpp/filesystem/create_directory
// https://stackoverflow.com/questions/6926433/how-to-shuffle-a-stdvector

namespace fs = std::filesystem;
using namespace std;

// checks if a directory exists
int dirExists(const char* const path)
{
    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? 1 : 0;
}

// saves all the names of the files in a given path
vector<string> getFiles(string path){

        //cout << "adding " + path << endl;
        // this is it for now
        vector<string> files;

        for(const auto & entry : fs::directory_iterator(path)){
                string it = entry.path();

                //cout << it << endl;
                files.push_back(it);
        }

        return files;


}

bool createOutputDirs(string outputPath){

	// three directories to save all the augmented data
	string masks = outputPath + "masks/";
	string imgs = outputPath + "imgs/";
	string xml = outputPath + "xml/"; 



	if (!dirExists(outputPath.c_str())){
	
		cout << "> Label: " << outputPath << " does not exist." << endl;

		if(fs::create_directory(outputPath)){
			cout << "> Label: created " << outputPath << endl;
		}else{
		
			cout << "> Label: could not create " << outputPath;
			throw "Error: Could not create output path.";
		}
	
	}

	if(dirExists(masks.c_str())){
		cout << "> Output for masks/ already exists, using existing directory" << endl;
	}else{

		if(fs::create_directory(masks)){
			cout << "> Label: created " << masks << endl;
		}else{
		
			cout << "> Label: could not create " << masks << endl;
			throw "Error: Could not create masks path.";
		}

	}

	if(dirExists(imgs.c_str())){
		cout << "> Output for imgs/ already exists, using existing directory" << endl;
	}else{

		if(fs::create_directory(imgs)){
			cout << "> Label: created " << imgs << endl;
		}else{
		
			cout << "> Label: could not create " << imgs << endl;
			throw "Error: Could not create imgs path.";
		}

	}

	if(dirExists(xml.c_str())){
		cout << "> Output for xml/ already exists, using existing directory" << endl;
	}else{

		if(fs::create_directory(xml)){
			cout << "> Label: created " << xml << endl;
		}else{
		
			cout << "> Label: could not create " << xml << endl;
			throw "Error: Could not create xml path.";
		}

	}
	
	return true;



}

vector<string> getInputs(string inputPath){

	vector<string> inputs;

	if (dirExists(inputPath.c_str())){
       		inputs = getFiles(inputPath);
       	}else{
                cout << "> Label: error could not find path " << inputPath << endl;
                throw "> Error: Invalid input path.";
        }

	return inputs;
}

vector<string> getBackgrounds(string backgroundPath){

	vector<string> backgrounds;

	if (dirExists(backgroundPath.c_str())){
       		backgrounds = getFiles(backgroundPath);
       	}else{
                cout << "> Label: error could not find path " << backgroundPath << endl;
                throw "> Error: Invalid background image path.";
        }

	// shuffle the background images around 
	auto rng = default_random_engine {};
	shuffle(begin(backgrounds), end(backgrounds), rng);

	return backgrounds;


}


int main(int argc, char** argv){

	vector<string> inputs;
	vector<string> backgrounds;
	string inputPath = "/home/andrew/Projects/School/os/";
	string outputPath = "/home/andrew/Projects/Projects/data/test/";
	string backgroundPath = "/home/andrew/Programs/Tensorflow/augmentation/randomShapes/blurry-backgrounds";

	inputs = getInputs(inputPath);
	cout << inputs.size() << endl;

	backgrounds = getBackgrounds(backgroundPath);
	cout << backgrounds.size() << endl;

	for (int i = 0; i < 35; i ++){
		cout << backgrounds.at(i) <<endl;
	}

	bool worked = createOutputDirs(outputPath);


	return 0;
}

