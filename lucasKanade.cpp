
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <zbar.h>

namespace convert
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

std::vector<cv::Point2f> PointsToTrack(int imageLabel)
{
		
		std::string fileLabel = convert::to_string(imageLabel);
		std::string fileName  = "/home/umairakmal/Desktop/OpticalFlow/GaborFeaturePoints/featurePoints_" + fileLabel + ".csv";

		std::ifstream inputfile (fileName.c_str());
		std::string current_line,csvItem;
		// vector allows you to add data without knowing the exact size beforehand
		std::vector<cv::Point2f> featurePoints;
		// Start reading lines as long as there are lines in the file
		while(inputfile.good())
		{			
			getline(inputfile, current_line);
			//imageName.push_back(current_line);
			//std::cout<<current_line<<std::endl;
			std::stringstream  lineStream(current_line);
    		std::string        cell;
        	// You have a cell!!!!
        	cv:: Point coordinatePoints;
        	std::getline(lineStream,cell,',');
        	//std::cout<<cell<<std::endl;
        	coordinatePoints.x = atof(cell.c_str());
        	std::getline(lineStream,cell,',');
        	//std::cout<<cell<<std::endl;
        	coordinatePoints.y = atof(cell.c_str());
        	//std::cout<<"data type "<<typeid(coordinatePoints.y).name()<<std::endl;
			featurePoints.push_back(coordinatePoints);
        	

		}

		return featurePoints;
} 
std::vector<cv::Point2f> RemoveJunkPoints(std::vector<cv::Point2f> & featurePoints)
{
	std::vector<cv::Point2f> junkFreeFeaturePoints;
	for (int i=0;i<featurePoints.size();i++)
	{
		if (featurePoints[i].x != 0.0 & featurePoints[i].y != 0.0)  
		{
			junkFreeFeaturePoints.push_back(featurePoints[i]);
		}

	}
	return junkFreeFeaturePoints;

}


cv::Mat LoadImage(const std::string & imagePath)
{
    // load and convert image
    cv::Mat image = cv::imread( imagePath, CV_LOAD_IMAGE_UNCHANGED );
    if ( !image.data )
    {
        throw 20;
    }

    if(image.channels() != 1)
    {
    	cv::cvtColor(image,image,CV_BGRA2GRAY);
    	
    }
    //std::cout<<"image type " <<image.channels() << std::endl;
    return image;
}
float Median(std::vector<float> & scores)
{
  float median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0)
  {
      median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  }
  else 
  {
      median = scores[size / 2];
  }

  return median;
}
void FindMeanEuclideanDistance(std::vector<cv::Point2f> & gaborFeaturePoints,std::vector<cv::Point2f> & lucasFeaturePoints, float & xOffsetTrackedPoints, float & yOffsetTrackedPoints, float & euclideanDistance)
{
	std::vector<float> temp_xOffsetTrackedPoints;
	std::vector<float>  temp_yOffsetTrackedPoints;
	std::vector<float>  temp_euclideanDistance;
	const int pointDisplacementThres = 20;
	std::vector<int> lostFeaturePointsIndices;
	for (int i=0;i<gaborFeaturePoints.size();i++)
	{ 
		if ((gaborFeaturePoints[i].y - lucasFeaturePoints[i].y) > pointDisplacementThres)
		{
			lostFeaturePointsIndices.push_back(i);
			gaborFeaturePoints[i] = cvPoint(0,0);
			
		}
		else
		{

			temp_xOffsetTrackedPoints.push_back(gaborFeaturePoints[i].x - lucasFeaturePoints[i].x);
			temp_yOffsetTrackedPoints.push_back(gaborFeaturePoints[i].y - lucasFeaturePoints[i].y);
			temp_euclideanDistance.push_back(sqrt(pow(temp_xOffsetTrackedPoints[i],2) + pow(temp_yOffsetTrackedPoints[i],2)));
		}	

	}
	xOffsetTrackedPoints = Median(temp_xOffsetTrackedPoints);
	yOffsetTrackedPoints = Median(temp_yOffsetTrackedPoints);
	euclideanDistance = Median(temp_euclideanDistance);

}
int main(int argc, char** argv)
{
	try
	{  
		// Reading Images names from CSV
		std::ifstream inputfile ("/home/umairakmal/Desktop/OpticalFlow/ImageNames.csv");
		std::string current_line;
		std::vector<std::string> imageName;// vector allows you to add data without knowing the exact size beforehand
		while(inputfile.good()) // Start reading lines as long as there are lines in the file
		{			
			getline(inputfile, current_line);
			std::string name = current_line.substr (0,current_line.length()-1);
			imageName.push_back(name);
		}
		//
		// Reading Index number of Car Camera Image
		std::ifstream inputfile1 ("/home/umairakmal/Desktop/OpticalFlow/idxCar.csv");
		std::string current_line1;
		std::vector<int> imageIndex;// vector allows you to add data without knowing the exact size beforehand
		while(inputfile1.good()) // Start reading lines as long as there are lines in the file
		{			
			getline(inputfile1, current_line1);
			std::string index = current_line1.substr (0,current_line1.length()-1);
			imageIndex.push_back(atoi(index.c_str()));
			
		}

		//

		std::string imageDirectoryPath = "/media/umairakmal/INTENSO/unverschluesselt/LB-EE_326_20170201_161537_split_000_dynamic_run2_rec/";		
		std::string imagePath = imageDirectoryPath + imageName[0];
		std::cout<<imagePath<<std::endl;
		cv:: Mat firstFrame = LoadImage(imagePath); // reads image and return a gray image
		cv::imshow("First Frame!",firstFrame);

		// Initial track Points
		std::vector<cv::Point2f> featuresPointsToTrack; 
		std::vector<cv::Point2f> allFeaturesPointsToTrack; 
		allFeaturesPointsToTrack = PointsToTrack(1);
		featuresPointsToTrack = RemoveJunkPoints(allFeaturesPointsToTrack);// removes zero points 
		//
		// Lucas-kanade Parametrs
		int lucas_window_size = 41;
		int numberOfPyramids = 5;
		cv:: Mat prevImg = firstFrame;
		std::vector<cv::Point2f> nextPts; 
		std::vector<uchar> status;
		std::vector<float>  err;
		int myradius=5;
		//
		// Image visualization parameters
		int thickness = 2;
		int lineType = 8;
		int shift = 0;
		// For text
		double textFontScale = 0.5;
		int textThickness = 1; 
		int textLineType = 8; 
		bool textBottomLeftOrigin = false; 
		//
		// Frame interval setting after which lucas-kanade points are replaced by Gabor or after which Gbaor comes into play
		int gaborFrame = 1;
		int gaborFeatureFrameSkip = 10;
		//
		// Euclidean Parameters
		float xOffsetTrackedPoints = 0;
		float yOffsetTrackedPoints = 0;
		float euclideanDistance = 0;
		//
        // Tracking loop on Image series
		for (int i=1; i < (imageName.size()) ; i++)
		{	
			std::string imagePath = imageDirectoryPath + imageName[i];
			cv:: Mat currentFrame = LoadImage(imagePath); // load new image/frame , reads image and return a gray image
			
			calcOpticalFlowPyrLK(prevImg,currentFrame,featuresPointsToTrack, nextPts, status, err,cv::Size( lucas_window_size, lucas_window_size ), numberOfPyramids,
		 						cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );// lucas-kanade implementation
			prevImg = currentFrame;
			cv::Mat visualImage = cv::imread( imagePath, CV_LOAD_IMAGE_UNCHANGED ); // reads RGB image for Visualization
			std::cout << i <<" "<< imageIndex[gaborFrame] << std::endl;
			if (i == imageIndex[gaborFrame])
			{	

				
				allFeaturesPointsToTrack = PointsToTrack(gaborFrame);
				gaborFrame=gaborFrame+(1+gaborFeatureFrameSkip);

				FindMeanEuclideanDistance (featuresPointsToTrack, nextPts, xOffsetTrackedPoints, yOffsetTrackedPoints, euclideanDistance);
				featuresPointsToTrack = RemoveJunkPoints(allFeaturesPointsToTrack);
				// Visualization when Gabor Comes into play
				for (int i=0;i<nextPts.size();i++)
				{	
					rectangle(visualImage, cvPoint(featuresPointsToTrack[i].y,featuresPointsToTrack[i].x), cvPoint(featuresPointsToTrack[i].y+2,featuresPointsToTrack[i].x+2), CV_RGB(250,0,0), thickness, lineType, shift=0);
				} // end Visualization loop
			}
			else
			{
				// Visualization when lucas-kanade is running
				for (int i=0;i<nextPts.size();i++)
				{	
					rectangle(visualImage, cvPoint(nextPts[i].y,nextPts[i].x), cvPoint(nextPts[i].y+2,nextPts[i].x+2), CV_RGB(0,0,250), thickness, lineType, shift=0);
				} // end Visualization loop

			}

			// Visualization of image sequence with points			
			putText(visualImage,"Frame Count   " + convert::to_string(i-1),cvPoint(20,150), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			putText(visualImage,"Frame to Reinitialization   " + convert::to_string(i-imageIndex[gaborFrame]), cvPoint(20,170), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			putText(visualImage,"Number of Features   " + convert::to_string(featuresPointsToTrack.size()), cvPoint(20,190), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			putText(visualImage,"Overall Euclidean Displacement   " + convert::to_string(euclideanDistance), cvPoint(20,210), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			putText(visualImage,"Overall x Displacement   " + convert::to_string(xOffsetTrackedPoints), cvPoint(20,230), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			putText(visualImage,"Overall y Displacement   " + convert::to_string(yOffsetTrackedPoints), cvPoint(20,250), cv::FONT_HERSHEY_SIMPLEX, textFontScale, CV_RGB(250,250,0), textThickness, textLineType, textBottomLeftOrigin);
			cv::imshow("Current Frame!",visualImage);
			cv:: waitKey(200);
			//

		} // end tracking loop		

	} // end try
	catch (int e)
	{
		if (e == 20)
		{
			std::cout<<"No image found for give path!"<<std::endl;
		}

	} // end catch

	return 0;
}
// --- EOF ---------------------------------------------------------------------