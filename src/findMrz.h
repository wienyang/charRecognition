#pragma once
#include <opencv2\opencv.hpp>
#include <string>
#include "utils.h"
#include "config.h"




//去除离群轮廓
int removeOutlierContours(cv::Mat &img);

//type 0 标准大护照 1 港澳通行证卡牌 2 港澳回乡证
int findMrz(const cv::Mat& img, int type, std::string& outDir);

int findMrz(std::string& imgPath, int type, std::string& outDir);

cv::Mat findMrz(const cv::Mat& img, int type);