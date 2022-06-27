#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <random>


using namespace std;
using namespace cv;


template<typename T>
T random(T from, T to) {
        random_device                    rand_dev;
        mt19937                          generator(rand_dev());
        uniform_real_distribution<T>    distr(from, to);
	return distr(generator);
}


int main() {


	Mat src = imread("/home/andrew/Pictures/profile.png");

	// rows = height and cols = width


	float randVal1 = random<float>(0.20f, 0.45f);
	float randVal2 = random<float>(0.70f, 0.90f);
	float randVal3 = random<float>(0.10f, 0.35f);
	float inverted = random<float>(0.0f, 1.0f);


	// print random value for troubleshooting
	//cout << "value1: " + to_string(randVal1) << endl;	
	//cout << "value2: " + to_string(randVal2) << endl;
	//cout << "value3: " + to_string(randVal3) << endl;

	// image size for troubleshooting
	//cout << "col: " + to_string(src.cols) << endl;
        //cout << "rows: " + to_string(src.rows) << endl;

	// points2f means a 2d set of floating point single precision vatiables
	// source
    	Point2f srcTri[3];

	if (inverted >= 0.5f){
    		srcTri[0] = Point2f( 0.f, 0.f ); // top left
    		srcTri[1] = Point2f( src.cols - 1.f, 0.f ); // bottom left
   		srcTri[2] = Point2f( 0.f, src.rows - 1.f ); // top right
	}else{
        	srcTri[0] = Point2f( src.cols - 1.f, src.rows - 1.f ); // bottom right
        	srcTri[1] = Point2f( 0.f, src.rows - 1.f ); // top right
        	srcTri[2] = Point2f( src.cols - 1.f, 0.f ); // bottom left
	}

	//destination
    	Point2f dstTri[3];
    	dstTri[0] = Point2f( 0.f, src.rows*randVal3 );
    	dstTri[1] = Point2f( src.cols*0.85f, src.rows*randVal1 );
    	dstTri[2] = Point2f( src.cols*0.15f, src.rows*randVal2 );

	// create a wrap matrix
    	Mat warp_mat = getAffineTransform( srcTri, dstTri );
    	Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
    	warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

    	imshow( "Warp", warp_dst );
    	waitKey();
    	return 0;


}
