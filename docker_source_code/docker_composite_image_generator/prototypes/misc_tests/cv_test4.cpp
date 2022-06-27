#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

using namespace std;


// goal load a video file and process it with multithearding

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


cv::Mat blackWhite(cv::Mat img){

	cv::Mat grey;
	cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
	return grey;
	


}


uint64_t timeSinceEpochMillisec(){
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}





int main(int argc, const char* argv[]){

        string path = "/mnt/0493db9e-eabd-406b-bd32-c5d3c85ebb38/Projects/Video/Weeds2/nenuphare/data1595119927.9510028output.avi";
	string save = "/mnt/0493db9e-eabd-406b-bd32-c5d3c85ebb38/Projects/dump/";
	
	int64_t  start = timeSinceEpochMillisec();

	cout <<  start << endl;

	cv::VideoCapture cap(path);



	cv::Mat frame;




    	while (1){

		if (!cap.read(frame)){

			cout << "al done" << endl;
                        break;

		}
 
                //cout << "test" << endl; 

		cv::Mat img = blackWhite(frame);

                //cout << "OK" << endl;

                //string windowName = "display " + to_string(i);
                 int64_t  current = timeSinceEpochMillisec();

                 string name = save + to_string(current) + "ok.jpg";
                 cv::imwrite(name, img);

            	// this code will be a task that may be executed immediately on a different core or deferred for later execution

    	} // end of while loop and single region




	uint64_t  end = timeSinceEpochMillisec();

	uint64_t total = end - start;

	cout << end << endl;
	cout << total << endl;

	return 0;
}
