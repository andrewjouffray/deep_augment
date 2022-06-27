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

cv::Mat lowerRes(cv::Mat image){

	// reduction factor
	float factor = 0.0;


	int height = image.rows;
	int width = image.cols;
	//allows the resolution do be reduced more on larger images
	if (height > 720){
		factor = randomFloat(1.0, 2.0);
	}else{
		factor = randomFloat(1.0, 1.5);
	}

	float newHeightf = height/factor;
	float newWidthf = width/factor;
	int newWidth = (int)newWidthf;
	int newHeight = (int)newHeightf;

	cout << "size ";
	cout << newWidth << endl;

	cv::Mat lowRes;
	cv::resize(image, lowRes, cv::Size(newWidth, newHeight));
	cv::resize(lowRes, image, cv::Size(width, height));

	return image;

}

cv::Mat blurr(cv::Mat image){

	int kernelSize = randomInt(1, 5);

	if(kernelSize % 2 == 0){
		kernelSize = kernelSize + 1;
	}
		
	cv::GaussianBlur(image, image, cv::Size(kernelSize, kernelSize), 0);

	return image;

}

int main(){

	cv::Mat objectImage = cv::imread("../selfie.png");


	cout << "resising an image" << endl;

	cv::resize(objectImage, objectImage, cv::Size(300, 300));

	cout << "creating an image" << endl;

	imshow("object",objectImage);
	int k = cv::waitKey(0);


	cv::Mat canvas(600, 1200, CV_8UC3, cv::Scalar(0,0,0));

	cout << "putting the object in the image" << endl;

	// define x1, x2, y1, y2
	int x1 = 30;
	int y1 = 45;
	int x2 = objectImage.cols;
	int y2 = objectImage.rows;

	cv::Mat insetImage(canvas, cv::Rect(x1, y1, x2, y2));
	objectImage.copyTo(insetImage);

	//objectImage.copyTo(canvas(cv::Rect(x1,y1, x2, y2)));

	cout << "applying transformations" << endl;

	cv::Mat can1 = blurr(canvas.clone());
	cv::Mat can2 = lowerRes(canvas.clone());

	imshow("canvas",canvas);
	k = cv::waitKey(0);
	imshow("blurr", can1);
	k = cv::waitKey(0);
	imshow("res", can2);
	k = cv::waitKey(0);
	

	return 0;



}

