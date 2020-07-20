#include "rotate.h"
#include <vector>
#include <cmath>

#include <fstream>



using namespace std;
using namespace cv;



double getTheta(const cv::Mat& img)
{
	Mat imgCopy = img.clone();
	imgCopy = imgCopy(Rect(imgCopy.cols * 1 / 5, imgCopy.rows * 3 / 4, imgCopy.cols * 3 / 5, imgCopy.rows - imgCopy.rows * 4 / 5));
	otsuBinaryzation(imgCopy);
	Canny(imgCopy, imgCopy, 30, 300);

#ifdef SHOW
	namedWindow("r1", 0);
	imshow("r1", imgCopy);
	waitKey(0);
#endif // SHOW



	vector<pair<int, int>> pointsVec;
	for (int i = 0; i < imgCopy.rows; ++i)
		for (int j = 0; j < imgCopy.cols; ++j)
			if (imgCopy.at<uchar>(i, j) > 0)
				pointsVec.push_back(pair<int, int>(i, j));

	Mat points(pointsVec.size(), 2, CV_64FC1);
	for (int n = 0; n < pointsVec.size(); n++)
	{
		points.at<double>(n, 0) = (double)pointsVec[n].first;
		points.at<double>(n, 1) = (double)pointsVec[n].second;
	}


	PCA pca(points, Mat(), PCA::DATA_AS_ROW);
	double theta = atan(pca.eigenvectors.at<double>(0, 0) / pca.eigenvectors.at<double>(0, 1));


	theta = 180.0*theta / 3.1415926;
	return theta;
}

double getTheta(const std::string& imgPath)
{
	Mat img = imread(imgPath);
	return getTheta(img);
}

cv::Mat rotate(const cv::Mat& img, double theta)
{
	Mat originImg = img.clone();
	Mat imgCopy = img.clone();

#ifdef SHOW
	namedWindow("r2", 0);
	imshow("r2", imgCopy);
	waitKey(0);
#endif
	//¶þÖµ»¯
	//hsvBinaryzation(imgCopy, threshold);
	otsuBinaryzation(imgCopy);
	//adaptiveBinaryzation(imgCopy);
	Mat binImg = imgCopy.clone();
#ifdef SHOW
	namedWindow("r3", 0);
	imshow("r3", imgCopy);
	waitKey(0);
#endif // SHOW

	Point2f center(imgCopy.cols / 2, imgCopy.rows / 2);

	Mat rot = getRotationMatrix2D(center, theta, 1);
	Rect bbox = RotatedRect(center, imgCopy.size(), theta).boundingRect();

	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	int originHeight = originImg.rows;

	Mat rotatedImg;
	cv::warpAffine(originImg, rotatedImg, rot, bbox.size());

	int cutMargin = (rotatedImg.rows - originHeight) / 2;
	rotatedImg = rotatedImg(Rect(0, cutMargin, rotatedImg.cols, rotatedImg.rows - 2 * cutMargin - 1));

#ifdef SHOW
	namedWindow("r4", 0);
	imshow("r4", rotatedImg);
	waitKey(0);
#endif // SHOW

	return rotatedImg;
}

cv::Mat rotate(const std::string& imgPath, double theta)
{
	Mat img = imread(imgPath);
	return rotate(img, theta);
}