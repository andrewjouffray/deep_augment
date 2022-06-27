#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

using namespace std;




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

        string path = "/home/andrew/Pictures";
	
	int64_t  start = timeSinceEpochMillisec();


        vector<string> paths = getFiles(path);	

	cout <<  start << endl;





	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < 58; i ++){

			string path = paths.at(i);
			//cout << "thread getting " + path << endl;

			cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

			img = blackWhite(img);

			//cout << "OK" << endl;

			//string windowName = "display " + to_string(i);
			//
			
			string name = path + to_string(i) + "ok.jpg";
			cv::imwrite(name, img);

			//cv::imshow(windowName, img);
			//int k = cv::waitKey(0);

		}
	}

	uint64_t  end = timeSinceEpochMillisec();

	uint64_t total = end - start;

	cout << end << endl;
	cout << total << endl;

	return 0;
}
