#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>

//https://stackoverflow.com/questions/701648/create-xml-files-from-c-program

namespace fs = std::filesystem;
using namespace std;

vector<vector<string>> parseFile( )
{

	string line;
	ifstream myfile ("../config.yeet");
	vector<vector<string>> parsedFile;
	if (myfile.is_open()){
    		while ( getline (myfile,line) ){
      			//cout << line << '\n';
			
			std::string delimiter = " ";

			vector<string> splitLine;
			size_t pos = 0;
			std::string token;
			while ((pos = line.find(delimiter)) != std::string::npos) {
    				token = line.substr(0, pos);
				// check if empty char
				if (token.compare(" ") != 0 && token.compare("") != 0){
					splitLine.push_back(token);
				}
    				line.erase(0, pos + delimiter.length());
			}
			if (token.compare(" ") != 0){
				splitLine.push_back(line);
			}
			parsedFile.push_back(splitLine);
    		}
   	 myfile.close();
 	
	}
  	else{ cout << "Unable to open file";

	}

	return parsedFile;
}

// goes through each line and reads the config
void setSettings (vector<vector<string>> file){

	// all the int values represent percentages
	int obj_affineProb = 50;
	int obj_changeSatProb = 30;
	int can_changeBrightProb = 30;
	int can_blurrProb = 30;
	int mult = 30;
	int objects = 40;
	int can_lowerRes = 20;

	// paths and names of folders
	string pathToBackground;
	string output = "installDir/output/datasetName";
	string label;
	string dataset;

	for(vector<string> line : file){
		
		// get the first word of the line
		string word = line.at(0);
		
		if(word.compare("dataset_name") == 0){
			
			dataset = line.at(2);

		}else if(word.compare("label_name") == 0){
			
			label = line.at(2);
		}
		else if(word.compare("output_path") == 0){
			
			output = line.at(2);
		}
		else if(word.compare("background_path") == 0){
			
			pathToBackground = line.at(2);
		}
		else if(word.compare("max_objects_per_canvas") == 0){
			
			objects = stoi(line.at(2));
		}
		else if(word.compare("canvases_per_frame") == 0){
			
			mult = stoi(line.at(2));
		}
		else if(word.compare("canvas_blurr") == 0){
			
			can_blurrProb = stoi(line.at(2));
		}
		else if(word.compare("object_saturation") == 0){
			
			obj_changeSatProb = stoi(line.at(2));
		}
		else if(word.compare("canvas_lower_resolution") == 0){
			
			can_lowerRes = stoi(line.at(2));
		}
		else if(word.compare("canvas_change_brightness") == 0){
			
			can_changeBrightProb = stoi(line.at(2));
		}
		else if(word.compare("object_affine_transform") == 0){
			
			obj_affineProb = stoi(line.at(2));
		}
		else if(word.compare("//") == 0){
			
			//do nothing
		}
		

	}

	cout << "\n========================= Label Configuration ===================================" << endl;
	cout << "> readFile: path to background:                             " << pathToBackground << endl;
	cout << "> readFile: output path:                                    " << output << endl;
	cout << "> readFile: dataset name:                                   " << dataset << endl;
	cout << "> readFile: label name:                                     " << label << endl;
	cout << "> readFile: number of canvases created per video frame:     " << mult << endl;
	cout << "> readFile: max number of objects to be put in each canvas: " << objects << endl;
	cout << "\n========================= Canvas Modification ===================================" << endl;
	cout << "> readFile: chances of blurring canvas:                     " << can_blurrProb << "%" << endl;
	cout << "> readFile: chances of lowering the canvas resolution:      " << can_lowerRes << "%" << endl;
	cout << "> readFile: chances of changing the canvas brightness:      " << can_changeBrightProb << "%" << endl;
	cout << "\n========================= Object Modification ===================================" << endl;
	cout << "> readFile: chances of changing object color saturation:    " << obj_changeSatProb << "%" << endl;
	cout << "> readFile: chances of changing object affine transform:    " << obj_affineProb << "%" << endl;

}

int main(int argc, char** argv){

	vector<vector<string>> file = parseFile();

	setSettings(file);
	return 0;
}

