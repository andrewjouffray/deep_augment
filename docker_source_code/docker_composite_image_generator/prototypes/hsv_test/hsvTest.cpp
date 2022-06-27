#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <random>


using namespace std;
using namespace cv;


Mat img;

template<typename T>
T random(T from, T to) {
        random_device                    rand_dev;
        mt19937                          generator(rand_dev());
        uniform_int_distribution<T>    distr(from, to);
	return distr(generator);
}


int main() {


	Mat src = imread("/home/andrew/Pictures/profile.png");

	// rows = height and cols = width


	// HSV stands for Hue Saturation Value(Brightness)
   	cvtColor(src,img,CV_RGB2HSV);


	int defaultVal = 255;

	// less than 255 will reduce the current hue value
	// greater than 255 will increase the hue value
	
	int hueVal = random<int>(155, 355);
    	int hue = hueVal  - defaultVal;


	int satVal = random<int>(155, 355);
    	int saturation = satVal  - defaultVal;

	// value is brightness, we modify this on the whole canvas not on one object
    	int value = 255 - defaultVal;
 
    	for(int y=0; y<img.cols; y++)
    	{
        	for(int x=0; x<img.rows; x++)
        	{
            	int cur1 = img.at<Vec3b>(Point(y,x))[0];
            	int cur2 = img.at<Vec3b>(Point(y,x))[1];
            	int cur3 = img.at<Vec3b>(Point(y,x))[2];
            	cur1 += hue;
            	cur2 += saturation;
            	cur3 += value;
 
            	if(cur1 < 0) cur1= 0; else if(cur1 > 255) cur1 = 255;
            	if(cur2 < 0) cur2= 0; else if(cur2 > 255) cur2 = 255;
            	if(cur3 < 0) cur3= 0; else if(cur3 > 255) cur3 = 255;
 
            	img.at<Vec3b>(Point(y,x))[0] = cur1;
            	img.at<Vec3b>(Point(y,x))[1] = cur2;
            	img.at<Vec3b>(Point(y,x))[2] = cur3;
        	}
    	}
 
    	cvtColor(img,src,CV_HSV2RGB);
    	imshow( "image", src );

	waitKey(0);
    	return 0;

}
