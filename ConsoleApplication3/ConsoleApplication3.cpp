// ConsoleApplication3.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"

#include  <fstream>
#include "ml.hpp"
#include "objdetect.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

// Sort identified rectangles from left to right
bool sortLefttoRight(const Rect &lhs, const Rect &rhs) {
	return lhs.x < rhs.x;
}

int main(int argc, char** argv)
{
	string train;

	cout << "Would you like to re-train the HOG SVM? (Y/N) \n";
	cin >> train;

	if (train == "Y" || train == "y") {

		// Delete old training data
		ofstream clearfile;
		clearfile.open("../../ConsoleApplication3/maxVal.txt", std::ofstream::out | std::ofstream::trunc);
		clearfile.close();

		ofstream clearfile2;
		clearfile2.open("../../ConsoleApplication3/SVM2.txt", std::ofstream::out | std::ofstream::trunc);
		clearfile2.close();

		// Save training data
		vector<Mat> allimage;

		// Loop through all 35 numbers and letters
		for (int i = 0; i <= 35; i++) {

			// Read all JPG files
			char filenames2[128];

			if (i < 10) {
				sprintf_s(filenames2, "../../ConsoleApplication3/Training/0%d/*.jpg", i);
			}
			else {
				sprintf_s(filenames2, "../../ConsoleApplication3/Training/%d/*.jpg", i);
			}

			String folderpath = filenames2;
			vector<String> filenames;
			glob(folderpath, filenames);

			// For each training image data
			for (int j = 0; j < filenames.size(); j++)
			{

				// Read image
				Mat immat = imread(filenames[j]);
				resize(immat, immat, Size(64, 128));
				Mat gray;
				cvtColor(immat, gray, CV_BGR2GRAY);
				allimage.push_back(gray);
			}
		}

		HOGDescriptor *hogDesc = new HOGDescriptor();

		// Compute first descriptor
		std::vector<float> desc; //
		hogDesc->compute(allimage.at(0), desc);

		// Contains sample descriptors
		int featureSize = desc.size(); //
		int numberOfSamples = 350;
		cv::Mat samples(numberOfSamples, featureSize, CV_32FC1); //

		// Compute descriptor of all samples
		for (int j = 0; j < 350; j++) {
			hogDesc->compute(allimage.at(j), desc);

			// Fill with descriptor
			for (int i = 0; i < featureSize; i++) //
				samples.ptr<float>(j)[i] = desc[i]; //
		}

		// Create labels
		cv::Mat labels(numberOfSamples, 1, CV_32SC1); //
		for (int i = 0; i < 35; i++) {
			labels.rowRange((i * 10), ((i + 1) * 10)) = i;
		}

		double minVal, maxVal;
		Point minLoc, maxLoc;
		int mul = 1000;

		vector<double> max;

		// normalize values to be in the range from zero to 1000 by dividing by maximum value
		for (int j = 0; j < featureSize; j++) {
			minMaxLoc(samples.col(j), &minVal, &maxVal, &minLoc, &maxLoc);

			if (maxVal != 0)
				samples.col(j) = samples.col(j) / maxVal * mul;

			max.push_back(maxVal);

			//save maxVal for normalization
			ofstream myfile;
			myfile.open("../../ConsoleApplication3/maxVal.txt", std::ofstream::app);
			myfile << maxVal << "\n";
			myfile.close();
		}

		// Create SVM classifier
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create(); //
		svm->setType(cv::ml::SVM::C_SVC); //
		svm->setKernel(cv::ml::SVM::INTER); //

		// Prepare training data
		cv::Ptr<cv::ml::TrainData> trainingData = //
			cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels); //

		// Training
		svm->train(trainingData); //

		svm->save("../../ConsoleApplication3/SVM2.txt");
	}

	// Window containing car image
	cv::namedWindow("Detector");

	// For each car image
	for (int fileNum = 1; fileNum <= 25; fileNum++) {

		// Read the image
		std::string file;
		file = std::to_string(fileNum);

		cv::Mat image = cv::imread("../../ConsoleApplication3/Testing/license-plate" + file + ".jpg");

		if ((!image.data))
		{
			cout << "Could not open or find the image" << endl;
			return -1;
		}

		// Convert to greyscale
		Mat grayImage;
		cvtColor(image, grayImage, CV_BGR2GRAY);

		// Another containing the rectangles to be drawn on top for users to see what letter is being processed
		Mat grayImage2;
		cvtColor(image, grayImage2, CV_BGR2GRAY);

		// Use MSER to detect features
		Ptr<MSER> ptrMSER = MSER::create();
		vector<vector<Point> > points;
		vector<Rect> rects;
		ptrMSER->detectRegions(grayImage, points, rects);

		// Contains just tall rectangles and of those just the largest rectangle if there is more than one within 5 pixels
		vector<Rect> r;

		// Iterate through MSER initial detections
		vector<Rect>::iterator itr = rects.begin();
		vector<vector<Point> >::iterator itp = points.begin();
		for (; itr != rects.end(); ++itr, ++itp) {

			// Make sure it's a character-sized rectangle
			if (itr->height <= itr->width * 1.75 || itr->height > itr->width * 3.5) {
				continue;
			}

			// Largest rectangle within 5 pixels; set to the current rectangle
			Rect max = *itr;

			// Iterate again to see if there is another character-sized rectangle within 5 pixels
			for (vector<Rect>::iterator itr2 = rects.begin(); itr2 != rects.end(); itr2++) {
				if (
					itr2->height > itr2->width * 1.75
					&& abs(itr2->x - max.x) < 5
					&& abs(itr2->y - max.y) < 5
					) {

					// Determine which is the larger rectangle
					if (itr2->area() > max.area()) {
						max = *itr2;
					}
				}
			}

			// Find the max rectangle and insert it
			if (std::find(r.begin(), r.end(), max) == r.end()) {
				r.push_back(max);
			}
		}

		// Contains supposed character recangles 
		vector<Rect> licenceCharacters;

		// See if there is another character next to it
		int count = 0;

		// Iterate through the above filtered rectangles
		for (vector<Rect>::iterator i = r.begin(); i != r.end(); ++i) {

			// Iterate again to see if there are at least two similar rectangles side-by-side which makes these two candidates being on a licence plate
			for (vector<Rect>::iterator i2 = r.begin(); i2 != r.end(); ++i2) {

				// Identify the largest width and height of the two rectangles
				int largerWidth;
				int largerHeight;

				if (i2->width > i->width) {
					largerWidth = i2->width;
				}
				else {
					largerWidth = i->width;
				}

				if (i2->height > i->height) {
					largerHeight = i2->height;
				}
				else {
					largerHeight = i->height;
				}

				// If the two rectangles meet the following conditions, they are similar and side-by-side
				if (
					abs(i2->width - i->width) < largerWidth * 0.2
					&& abs(i2->height - i->height) < largerHeight * 0.2
					&& abs(i2->y - i->y) < largerHeight / 3
					&& abs(i2->x - i->x) < largerWidth * 3
					&& abs(i2->x - i->x) > largerWidth * 0.8
					) {

					// Increment
					count++;
				}
			}

			// If there is at least two similar side-by-side rectangles insert them
			if (count > 0) {
				licenceCharacters.push_back(*i);

				// Reset counter for next rectangle to be checked
				count = 0;
			}
		}

		// Ignore following steps if no licence plate characters are identified
		if (!licenceCharacters.empty()) {

			// Sort rectangles from left to right
			sort(licenceCharacters.begin(), licenceCharacters.end(), sortLefttoRight);

			// See if there needed to be additional height since characters will be displayed to the left vertically and add height if necessary
			int height = grayImage.size().height;
			Mat additionalHeight;
			if (height <= licenceCharacters.size() * 60) {
				int addHeight = licenceCharacters.size() * 60 + 1 - height;

				resize(grayImage, additionalHeight, Size(grayImage.size().width, addHeight));
				additionalHeight = Scalar(0);

				vconcat(grayImage2, additionalHeight, grayImage2);

				height += addHeight;
			}

			// Black area to display characters vertically
			Mat side;
			resize(grayImage, side, Size(30, height));
			side = Scalar(0);

			// Display initial picture 
			Mat result;
			hconcat(side, grayImage2, result);
			cv::imshow("Detector", result);
			cv::waitKey(10);

			// Print out the licence plate as text
			cout << "\nLicence Plate = ";

			// Contains cropped characters vertically aligned
			Mat crop;

			// Iterate through all the suspected licence plate characters
			for (vector<Rect>::iterator i = licenceCharacters.begin(); i != licenceCharacters.end(); ++i) {

				// Resize the character to fit nicely in the vertical area
				Mat resized;
				resize(grayImage(*i), resized, Size(30, 60));

				//rectangle(grayImage, *i, CV_RGB(0, 255, 0));  //??????????????
				rectangle(grayImage2, *i, CV_RGB(0, 255, 0)); //??

				// For each iteration of loop, characters will be added to the vertical area
				if (i == licenceCharacters.begin()) {
					crop = resized;
				}
				else {
					vconcat(crop, resized, crop);
				}

				// Use SVM to determine the text of the character image
				Mat gray;
				resize(resized, gray, Size(64, 128));
				vector<float> desc;
				HOGDescriptor *hogDesc = new HOGDescriptor();
				hogDesc->compute(gray, desc);

				int featureSize = desc.size();
				Mat input(1, featureSize, CV_32FC1);
				vector<double> maxVal(featureSize);
				ifstream alldata;

				alldata.open("../../ConsoleApplication3/maxVal.txt", std::ofstream::in);

				for (int j = 0; j < featureSize; j++) {
					alldata >> maxVal[j];
				}

				for (int i = 0; i < featureSize; i++) {
					input.col(i) = desc[i] / maxVal.at(i) * 1000;
				}

				char output;

				Ptr<SVM> svmNew = SVM::create();
				svmNew = SVM::load("../../ConsoleApplication3/svm2.txt");

				output = svmNew->predict(input);

				switch (output)
				{
				case 0:
					output = '0';
					break;
				case 1:
					output = '1';
					break;
				case 2:
					output = '2';
					break;
				case 3:
					output = '3';
					break;
				case 4:
					output = '4';
					break;
				case 5:
					output = '5';
					break;
				case 6:
					output = '6';
					break;
				case 7:
					output = '7';
					break;
				case 8:
					output = '8';
					break;
				case 9:
					output = '9';
					break;
				case 10:
					output = 'A';
					break;
				case 11:
					output = 'B';
					break;
				case 12:
					output = 'C';
					break;
				case 13:
					output = 'D';
					break;
				case 14:
					output = 'E';
					break;
				case 15:
					output = 'F';
					break;
				case 16:
					output = 'G';
					break;
				case 17:
					output = 'H';
					break;
				case 18:
					output = 'I';
					break;
				case 19:
					output = 'J';
					break;
				case 20:
					output = 'K';
					break;
				case 21:
					output = 'L';
					break;
				case 22:
					output = 'M';
					break;
				case 23:
					output = 'N';
					break;
				case 24:
					output = 'O';
					break;
				case 25:
					output = 'P';
					break;
				case 26:
					output = 'Q';
					break;
				case 27:
					output = 'R';
					break;
				case 28:
					output = 'S';
					break;
				case 29:
					output = 'T';
					break;
				case 30:
					output = 'U';
					break;
				case 31:
					output = 'V';
					break;
				case 32:
					output = 'W';
					break;
				case 33:
					output = 'X';
					break;
				case 34:
					output = 'Y';
					break;
				case 35:
					output = 'Z';
					break;
				default:
					output = '?';
					break;
				}

				// Height of black vertical area shrinks each time a character is added to that area
				height -= 60;
				resize(grayImage, side, Size(30, height));
				side = Scalar(0);

				// Display updated result
				vconcat(crop, side, side);
				hconcat(side, grayImage2, result);
				cv::imshow("Detector", result);
				cv::waitKey(10);

				// Print SVM result or the character as text
				cout << output;
			}
		}
	}

	return 0;
}