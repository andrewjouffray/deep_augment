#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <sys/stat.h>
#include "./libs/tinyxml/tinyxml.h"
#include "./libs/tinyxml/tinystr.h"

//https://stackoverflow.com/questions/701648/create-xml-files-from-c-program

namespace fs = std::filesystem;
using namespace std;

void build_simple_doc( )
{

	// dummy vectors
	vector<int> roi1(4, 10);
	vector<int> roi2(4, 6);
	vector<int> roi3(4, 23);

	// dummy vector in the format of the getRois method
	vector<vector<int>> rois;
	rois.push_back(roi1);
	rois.push_back(roi2);
	rois.push_back(roi3);

	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	TiXmlElement * annotation = new TiXmlElement( "annotation" );
	TiXmlElement * folder = new TiXmlElement( "folder" );
	TiXmlElement * filename = new TiXmlElement( "filename" );
	TiXmlElement * path = new TiXmlElement( "path" );
	TiXmlElement * source = new TiXmlElement( "source" );
	TiXmlElement * database = new TiXmlElement( "database" );
	TiXmlElement * size = new TiXmlElement( "size" );
	TiXmlElement * width = new TiXmlElement( "width" );
	TiXmlElement * height = new TiXmlElement( "height" );
	TiXmlElement * depth = new TiXmlElement( "depth" );
	TiXmlElement * segmented = new TiXmlElement( "segmented" );


	TiXmlText * folderName = new TiXmlText( "masks-test" );
	TiXmlText * fileNameText = new TiXmlText( "fileName.jpg" );
	TiXmlText * databaseName = new TiXmlText( "Unknown" );
	TiXmlText * pathText = new TiXmlText( "/home/andrew/this/that/here/there" );
	TiXmlText * widthVal = new TiXmlText( "736" );
	TiXmlText * heightVal = new TiXmlText( "736" );
	TiXmlText * depthVal = new TiXmlText( "3" );
	TiXmlText * segmentedVal = new TiXmlText( "0" );


	annotation->LinkEndChild( folder );
		folder->LinkEndChild(folderName);

	annotation->LinkEndChild( filename );
		filename->LinkEndChild(fileNameText);

	annotation->LinkEndChild( path );
		path->LinkEndChild(pathText);

	annotation->LinkEndChild( source );
		source->LinkEndChild( database );
			database->LinkEndChild( databaseName );

	annotation->LinkEndChild( size );
		size->LinkEndChild( width );	
			width->LinkEndChild( widthVal );
		size->LinkEndChild( height );
			height->LinkEndChild( heightVal );
		size->LinkEndChild( depth );	
			depth->LinkEndChild( depthVal );

	annotation->LinkEndChild( segmented );
		segmented->LinkEndChild( segmentedVal );

	// add all the objects here might need to create them in the loop
	for(vector<int> roi : rois){

		TiXmlElement * object = new TiXmlElement( "object" );
		TiXmlElement * name = new TiXmlElement( "name" );
		TiXmlElement * pose = new TiXmlElement( "pose" );
		TiXmlElement * truncated = new TiXmlElement( "truncated" );
		TiXmlElement * difficult = new TiXmlElement( "difficult" );
		TiXmlElement * bndbox = new TiXmlElement( "bndbox" );
		TiXmlElement * xmin = new TiXmlElement( "xmin" );
		TiXmlElement * ymin = new TiXmlElement( "ymin" );
		TiXmlElement * xmax = new TiXmlElement( "xmax" );
		TiXmlElement * ymax = new TiXmlElement( "ymax" );

		TiXmlText * objectName = new TiXmlText( "person" );
		TiXmlText * objectPose = new TiXmlText( "Unspecified" );
		TiXmlText * objectTruncated = new TiXmlText( "0" );
		TiXmlText * objectDifficult = new TiXmlText( "0" );

		string x1 = to_string(roi[0]);
		string y1 = to_string(roi[1]);
		string x2 = to_string(roi[2]);
		string y2 = to_string(roi[3]);

		TiXmlText * objectXmin = new TiXmlText( x1.c_str() );
		TiXmlText * objectYmin = new TiXmlText( y1.c_str() );
		TiXmlText * objectXmax = new TiXmlText( x2.c_str() );
		TiXmlText * objectYmax = new TiXmlText( y2.c_str() );


	
		annotation->LinkEndChild( object );
			object->LinkEndChild(name);
				name->LinkEndChild(objectName);
			object->LinkEndChild(pose);
				pose->LinkEndChild(objectPose);
			object->LinkEndChild(truncated);
				truncated->LinkEndChild(objectTruncated);
			object->LinkEndChild(difficult);
				difficult->LinkEndChild(objectDifficult);
			object->LinkEndChild(bndbox);
				bndbox->LinkEndChild(xmin);
					xmin->LinkEndChild(objectXmin);
				bndbox->LinkEndChild(ymin);
					ymin->LinkEndChild(objectYmin);
				bndbox->LinkEndChild(xmax);
					xmax->LinkEndChild(objectXmax);
				bndbox->LinkEndChild(ymax);
					ymax->LinkEndChild(objectYmax);


	}

	



	doc.LinkEndChild( decl );
	doc.LinkEndChild( annotation );
	doc.SaveFile( "labelImgDemo.xml" );
}

int main(int argc, char** argv){

	build_simple_doc();

	return 0;
}

