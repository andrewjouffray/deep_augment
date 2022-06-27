#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include "randomFunc.h"

using namespace std;
int thresh = 100;

// this tests the radom functions, lowe res, blurr, create canvas and inserting an image onto the canvas

vector<vector<cv::Point>> get_rois(cv::Mat image){


	cv::Mat cloneMask;
	cv::cvtColor(image.clone(), cloneMask, cv::COLOR_BGR2GRAY);
	double thresh = 0;
	double maxValue = 255;
	// Binary Threshold
	cv::threshold(cloneMask, cloneMask, thresh, maxValue, cv::THRESH_BINARY);

	cv::Mat canny_output;
	cv::Canny(cloneMask, canny_output, thresh, thresh*2);	

    	vector<vector<cv::Point>> contours;
	cv::findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


	return contours;

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
	int x1 = randomInt(30, 300);
	int y1 = randomInt(30, 300);
	int x2 = objectImage.cols;
	int y2 = objectImage.rows;

	cv::Mat insetImage(canvas, cv::Rect(x1, y1, x2, y2));
	objectImage.copyTo(insetImage);

	int x3 = randomInt(300, 600);
	int y3 = randomInt(30, 300);

	cv::Mat insetImage2(canvas, cv::Rect(x3, y3, x2, y2));
	objectImage.copyTo(insetImage2);


	cout << "applying transformations" << endl;

	vector<vector<cv::Point>> contours = get_rois(canvas.clone());

	vector<vector<cv::Point> > contours_poly( contours.size() );
    	vector<cv::Rect> boundRect( contours.size() );

    	for( size_t i = 0; i < contours.size(); i++ )
    	{
		cout << "iterating over the countours" << endl;

		int areax = cv::contourArea(contours[i]);

		cout << "Area: " << areax << endl;

		// 3000 is a dummy value, best to calculate average
		if (areax > 3000){
			cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
        		boundRect[i] = cv::boundingRect( contours_poly[i] );
			cv::rectangle( canvas, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(255, 255, 255), 2 );

		}
    	}

	imshow("canvas",canvas);
	k = cv::waitKey(0);
	

	return 0;



}

