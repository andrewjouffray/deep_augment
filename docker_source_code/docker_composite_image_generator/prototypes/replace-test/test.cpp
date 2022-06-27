#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include "randomFunc.h"

using namespace std;


// this tests the radom functions, lowe res, blurr, create canvas and inserting an image onto the canvas

cv::Mat replace(cv::Mat image, cv::Mat background){


	// resize the background to fit the canvas
	cv::resize(background, background, cv::Size(image.cols, image.rows));


	cv::Mat cloneMask;
	cv::cvtColor(image.clone(), cloneMask, cv::COLOR_BGR2GRAY);
	double thresh = 0;
	double maxValue = 255;
	// Binary Threshold
	cv::threshold(cloneMask, cloneMask, thresh, maxValue, cv::THRESH_BINARY);
	cv::cvtColor(cloneMask, cloneMask, cv::COLOR_GRAY2BGR);


	cv::imshow("image", image);
	int k = cv::waitKey(0);

	// you will need to perform arythmetic (addition) of the two images
	// you will need to find a way to add the two images only in the spots that matter 
	cv::Mat comb = cloneMask + background;

	comb.setTo(0, comb == 255); 
	cv::Mat comb2 =  comb + image;


	return comb2;

}

int main(){

	cv::Mat objectImage = cv::imread("../selfie.png");
	cv::Mat back = cv::imread("../bck.jpg");

	cout << "resising an image" << endl;
	cv::resize(objectImage, objectImage, cv::Size(300, 300));

	cout << "creating an image" << endl;
	imshow("object",objectImage);
	int k = cv::waitKey(0);
	cv::Mat canvas(600, 1200, CV_8UC3, cv::Scalar(0,0,0));

	cout << "putting the object in the image" << endl;
	// define x1, x2, y1, y2
	int x1 = randomInt(30, 300);
	int y1 = randomInt(30, 300);
	int x2 = objectImage.cols;
	int y2 = objectImage.rows;
	cv::Mat insetImage(canvas, cv::Rect(x1, y1, x2, y2));
	objectImage.copyTo(insetImage);

	cout << "applying transformations" << endl;

	canvas = replace(canvas.clone(), back);

	imshow("canvas",canvas);
	k = cv::waitKey(0);
	

	return 0;



}

