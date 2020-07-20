#include "findMrz.h"
#include "rotate.h"
#include <algorithm>


using namespace cv;
using namespace std;

//#define SHOW
int removeOutlierContours(cv::Mat &img)
{
	//-----------------------------------------------------------------------
	//C++接口的findContours在边界处似乎存在bug，所以首先应该对边界进行预处理
	//四个边界全部置为0
	for (int i = 0; i < img.rows; ++i)
	{
		img.at<uchar>(i, 0) = 0;
		img.at<uchar>(i, img.cols - 1) = 0;
	}
	for (int j = 0; j < img.cols; ++j)
	{
		img.at<uchar>(0, j) = 0;
		img.at<uchar>(img.rows - 1, j) = 0;
	}
	//-----------------------------------------------------------------------

	Mat imgCopy = img.clone();
#ifdef SHOW
	namedWindow("rc0", 0);
	imshow("rc0", imgCopy);
	waitKey(0);
#endif // SHOW
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	for (vector<Point> contour : contours)
	{
		Rect contourRect = boundingRect(contour);
		imgCopy(contourRect) = 255;
	}

#ifdef SHOW
	namedWindow("rc1", 0);
	imshow("rc1", imgCopy);
	waitKey(0);
#endif // SHOW

	findContours(imgCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<Rect> rects;
	vector<int> heights;
	for (vector<Point> contour : contours)
	{
		Rect contourRect = boundingRect(contour);
		rects.push_back(contourRect);
		heights.push_back(contourRect.height);
	}
	sort(heights.begin(), heights.end());
	double refHeight = (double)heights[heights.size() / 2];

	for (Rect rect : rects)
	{
		double ratio = (double)rect.height / refHeight;

		if (ratio > 1.3 || ratio < 0.7)
			img(rect) = 0;
	}
#ifdef SHOW
	namedWindow("rc2", 0);
	imshow("rc2", img);
	waitKey(0);
#endif // SHOW
	return 0;
}

int findMrz(const Mat& img, int type, std::string& outDir)
{
	if (img.empty())
		return 1;
	if (outDir.back() != '/')
		outDir.push_back('/');

	Mat imgCopy = img.clone();
	resize(imgCopy, imgCopy, Size(1140, 822));//2280,1645


	if (type == 0 || type == -1)	//常规护照 0时考虑旋转
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 3 / 4, imgCopy.cols, imgCopy.rows - imgCopy.rows * 3 / 4));
	else if (type == 1)
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 4 / 5, imgCopy.cols, imgCopy.rows - imgCopy.rows * 4 / 5));
	else if (type == 2)
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 1 / 2, imgCopy.cols, imgCopy.rows - imgCopy.rows * 1 / 2));

#ifdef SHOW
	namedWindow("imgCopy", 0);
	imshow("imgCopy", imgCopy);
	waitKey(0);
#endif
	imwrite(outDir+"mrzImg.bmp", imgCopy);
	return 0;
}

int findMrz(string& imgPath, int type, std::string& outDir)
{
	const Mat img = imread(imgPath);
	if (img.empty())
		return 1;
	return findMrz(img, type, outDir);
}


cv::Mat findMrz(const cv::Mat & img, int type)
{
	if (img.empty())
		return {};
	Mat imgCopy = img.clone();
	if(type!=2)resize(imgCopy, imgCopy, Size(1140, 822));//2280,1645
	else resize(imgCopy, imgCopy, Size(1140, 720));//2280,1645

	if (type == 0 || type == -1)	
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 3 / 4, imgCopy.cols, imgCopy.rows - imgCopy.rows * 3 / 4));
	else if (type == 1)
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 4 / 5, imgCopy.cols, imgCopy.rows - imgCopy.rows * 4 / 5));
	else if (type == 2)
		imgCopy = imgCopy(Rect(0, imgCopy.rows * 1 / 2, imgCopy.cols, imgCopy.rows - imgCopy.rows * 1 / 2));

	return imgCopy;
}
