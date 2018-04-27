
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "objdetect.hpp"
#include	<fstream>

using namespace std;
using namespace cv;

int main()
{

	//clear data in "maxVal.txt" in the begining
	ofstream clearfile;
	clearfile.open("maxVal.txt", std::ofstream::out | std::ofstream::trunc);
	clearfile.close();

	vector<Mat>  allimage;
	
	
	//loop all 35 classes
	for (int i = 0; i<35; i++)
	{

		//read all .jpg from "data" folder
		char filenames2[128];
		if (i<10)
			sprintf_s(filenames2, "data/0%d/*.jpg", i);		//for 01 to 09 (folder name)
		else
			sprintf_s(filenames2, "data/%d/*.jpg", i);		//for 10 to 34 (folder name)

		String folderpath = filenames2;
		vector<String> filenames;
		glob(folderpath, filenames);

			//loop for each image
		for (size_t j = 0; j<filenames.size(); j++)
		{

			//read image
			Mat immat = imread(filenames[j]);
			resize(immat, immat, Size(64, 128));
			Mat gray;
			cvtColor(immat, gray, CV_BGR2GRAY);
			allimage.push_back(gray);

		}

	}

	// create HOG discriptor 
	HOGDescriptor *hogDesc = new HOGDescriptor();

	vector<Point> locations3;
	// compute first descriptor
	std::vector<float> desc;
	hogDesc->compute(allimage.at(0), desc, Size(32, 32), Size(0, 0), locations3);

	// the matrix of sample descriptors
	int featureSize = desc.size();
	int numberOfSamples = 350;

	// create the matrix that will contain the samples HOG
	cv::Mat samples(numberOfSamples, featureSize, CV_32FC1);



	// compute descriptor of the all the samples
	for (int j = 0; j < 350; j++) {
		hogDesc->compute(allimage.at(j), desc, Size(32, 32), Size(0, 0), locations3);

		// fill with descriptor
		for (int i = 0; i < featureSize; i++)
			samples.ptr<float>(j)[i] = desc[i];

	}

	// Create the labels
	cv::Mat labels(numberOfSamples, 1, CV_32SC1);

	// labels of samples (0-34)
	for (int i = 0; i<35; i = i++)
	{
		labels.rowRange((i * 10), ((i + 1) * 10)) = i;
	}


	double	minVal, maxVal;
	Point	minLoc, maxLoc;
	int mul = 1000;
	
	vector<double> max;
	// normalize values to be in the range
	for (int j = 0; j < featureSize; j++) {
		minMaxLoc(samples.col(j), &minVal, &maxVal, &minLoc, &maxLoc);

		if (maxVal != 0)
			samples.col(j) = samples.col(j) / maxVal * mul;
		max.push_back(maxVal);
		
		//save maxVal 
		ofstream myfile;
		myfile.open("maxVal.txt", std::ofstream::app);
		myfile << maxVal << "\n";
		myfile.close();
	}
	
	// create SVM classifier
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::INTER);

	// prepare the training data
	cv::Ptr<cv::ml::TrainData> trainingData =
		cv::ml::TrainData::create(samples,
			cv::ml::SampleTypes::ROW_SAMPLE,
			labels);
	// SVM training
	svm->train(trainingData);
	svm->save("SVM2.txt");
	//svm->trainAuto( trainingData,10 );

	cout << "training complete" << endl;

	return 0;
}