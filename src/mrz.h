#pragma once
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>

#include "config.h"


static const char charTable[37] = {
	'0', '1', '2', '3', '4', '5', '6',
	'7', '8', '9', 'A', 'B', 'C', 'D',
	'E', 'F', 'G', 'H', 'I', 'J', 'K',
	'L', 'M', 'N', 'O', 'P', 'Q', 'R',
	'S', 'T', 'U', 'V', 'W', 'X', 'Y',
	'Z', '<' };



class MRZ {
public:

	MRZ(int type) :m_type(type) {}
	MRZ(const cv::Mat &img, int type);
	MRZ(const std::string &imgPath, int type);
	~MRZ();

public:
	bool initNet(std::string modelPath);
	void setImg(cv::Mat& img);
	void checkMrzStr();
	int segmentChars();
	void recognize();
	std::string mrzStr();


#ifdef _DEBUG
public:
#else
private:
#endif // _DEBUG

	void alpha2num(char& c);
	void num2alpha(char& c);

	cv::Mat m_mrzImg;
	std::vector<std::vector<cv::Mat>> m_charImgMat;
	std::string m_mrzStr;
	int m_type;
	cv::dnn::Net m_net;
};