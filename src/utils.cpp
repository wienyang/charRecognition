#include "utils.h"
#include <io.h>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <Windows.h>
#include <tchar.h>
#include <numeric>



using namespace std;
using namespace cv;






std::vector<int> getDiffVec(std::vector<int> vec)
{
	std::vector<int> res;
	for (int i = 1; i < vec.size(); ++i)
		res.push_back(res[i] - res[i - 1]);
	return res;
}


void getFiles(std::string path, std::vector<std::string>& files, bool recurse)
{
	//文件句柄  
	intptr_t hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if (recurse)
			{
				if ((fileinfo.attrib &  _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						getFiles(p.assign(path).append("/").append(fileinfo.name), files, recurse);
				}
				else
				{

					files.push_back(fileinfo.name);
				}
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

std::string randomStr()
{
	std::string res = "";
	const std::string CCH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	static std::default_random_engine e(time(0));
	static std::uniform_int_distribution<unsigned> u(0, CCH.size() - 1);
	for (int i = 0; i < 10; ++i)
	{
		int x = u(e);
		res.push_back(CCH[x]);
	}
	return res;
}

int adaptiveBinaryzation(cv::Mat & img)
{
	if (img.channels() == 3)
		cvtColor(img, img, COLOR_BGR2GRAY);
	int blockSize = 41;
	int constValue = 60;
	Mat dst;
	adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, blockSize, constValue);
	return 0;
}

int otsuBinaryzation(cv::Mat & img)
{
	if (img.channels() == 3)
		cvtColor(img, img, COLOR_BGR2GRAY);
	threshold(img, img, 0, 255, THRESH_BINARY | THRESH_OTSU);//大津法
	img = img < 100;
	return 0;
}

int hsvBinaryzation(cv::Mat &img, int threshold)
{
	//二值化
	vector<Mat> imgSplit;
	cvtColor(img, img, COLOR_BGR2HSV);
	split(img, imgSplit);
	equalizeHist(imgSplit[2], imgSplit[2]);
	equalizeHist(imgSplit[1], imgSplit[1]);
	merge(imgSplit, img);
	inRange(img, Scalar(0, 0, 0), Scalar(180, 255, threshold), img);
	return 0;
}

int binaryImgHorizontalChange(cv::Mat & img, std::vector<int>& horizontalChangeVec)
{
	uchar lastPix = 0;	//上一个像素
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) != lastPix)
				horizontalChangeVec[i]++;
			lastPix = img.at<uchar>(i, j);
		}
	return 0;
}

void showVecHist(std::vector<int> vec)
{
	vector<int> sortedVec(vec);
	sort(sortedVec.begin(), sortedVec.end());

	Mat histImg(sortedVec.back(), sortedVec.size(), CV_8UC1);

	for (int j = 0; j < histImg.cols; ++j)
		for (int i = 0; i < histImg.rows; ++i) {
			//hist
			if (i < vec[j])
				histImg.at<uchar>(histImg.rows - 1 - i, j) = 255;
			else
				histImg.at<uchar>(histImg.rows - 1 - i, j) = 0;

		}
	namedWindow("hist", 0);
	imshow("hist", histImg);
	waitKey(0);
}

int vecSum(std::vector<int> vec)
{
	int sum = (int)0;
	for (int element : vec) {
		sum += element;
	}
	return sum;
}

void vecHighValuePassFilter(std::vector<int>& vec, int threadhold)
{
	for (int &element : vec)
		if (element < threadhold)
			element = 0;
}

bool SegCompareByWidth(Seg a, Seg b)
{
	return a.getWidth() < b.getWidth();
}

bool SegCompareByHead(Seg a, Seg b)
{
	return a.getHead() < b.getHead();
}

std::vector<Seg> vec2Segs(std::vector<int> vec)
{
	std::vector<Seg> segs;
	bool inChar = false;
	int head = 0;
	int tail = 0;
	for (int i = 0; i < vec.size(); ++i)
	{
		if (!inChar && vec[i] > 0)
		{
			inChar = true;
			head = i;
			continue;
		}
		if (inChar && (vec[i] == 0 || i == vec.size() - 1))
		{
			inChar = false;
			tail = i;
			segs.push_back(Seg(head, tail));
			continue;
		}
	}
	sort(segs.begin(), segs.end(), SegCompareByWidth);
	return segs;
}

int segsFilter(std::vector<Seg>& segs, int threshold)
{
	//过滤之后留下的
	vector<Seg > remained;
	if (segs.size() < 1)
		return -1;
	for (Seg seg : segs)
		if (seg.getWidth() > threshold)
			remained.push_back(seg);
	//最后根据所在位置进行排序
	sort(remained.begin(), remained.end(), SegCompareByHead);
	segs.swap(remained);
	return 0;
}

int vecInSegPassFilter(std::vector<int>& vec, std::vector<Seg> segs)
{
	vector<int> tmp(vec.size(), 0);
	for (Seg seg : segs)
		std::swap_ranges(
			tmp.begin() + seg.getHead(),
			tmp.begin() + seg.getTail() + 1,
			vec.begin() + seg.getHead()
		);
	vec.swap(tmp);
	return 0;
}


HMODULE GetSelfModuleHandle()
{
	MEMORY_BASIC_INFORMATION mbi;
	return ((::VirtualQuery(GetSelfModuleHandle, &mbi, sizeof(mbi)) != 0) ? (HMODULE)mbi.AllocationBase : NULL);
}

std::string getCurrentProgramPath()
{

	TCHAR curDir[100] = { 0 };
	GetModuleFileName(GetSelfModuleHandle(), curDir, 100);
	(_tcsrchr(curDir, _T('\\')))[1] = 0; // 删除文件名，只获得路径

	int iLen = WideCharToMultiByte(CP_ACP, 0, curDir, -1, NULL, 0, NULL, NULL);
	char* chRtn = new char[iLen * sizeof(char)];
	WideCharToMultiByte(CP_ACP, 0, curDir, -1, chRtn, iLen, NULL, NULL);
	std::string str(chRtn);
	return str;
}

void vecNormalization(std::vector<double>& vec, int scale)
{
	int sum = std::accumulate(vec.begin(), vec.end(), 0);
	int mean = sum / vec.size();
	for (double &dist : vec)
	{
		dist -= mean;
		dist /= scale;
	}
}

double vecNorm(std::vector<double> vec, double N)
{
	double res = 0.0;
	for (double num : vec)
		res += pow(num, N);
	return res;
}

void contrastEnhancement(cv::Mat img)
{
}

bool CompareRectByY(cv::Rect rect1, cv::Rect rect2)
{
	return rect1.y < rect2.y;
}

bool CompareRectByX(cv::Rect rect1, cv::Rect rect2)
{
	return rect1.x < rect2.x;
}


