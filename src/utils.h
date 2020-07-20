#pragma once
#include <vector>
#include <string>
#include <opencv2\opencv.hpp>


//获取差分向量
std::vector<int> getDiffVec(std::vector<int> vec);

//获取目录下文件 
void getFiles(std::string path, std::vector<std::string>& files, bool recurse);

//生成随机序列
std::string randomStr();

//自适应局部二值化
int adaptiveBinaryzation(cv::Mat &img);

//大津二值化
int otsuBinaryzation(cv::Mat &img);

//HSV空间二值化 
int hsvBinaryzation(cv::Mat &img, int threshold);

//二值化图像每行像素变化次数
int binaryImgHorizontalChange(cv::Mat &img, std::vector<int> &horizontalChangeVec);


//显示vector对应直方图
void showVecHist(std::vector<int> vec);

//求vector之和

int vecSum(std::vector<int> vec);


//将vector中低于阈值的值过滤（置为0）

void vecHighValuePassFilter(std::vector<int> &vec, int threadhold);




class Seg {
public:
	Seg(int head, int tail) {
		this->head = head;
		this->tail = tail;
		this->width = tail - head;
	}
	int getWidth() {
		return this->width;
	}
	int getHead() {
		return this->head;
	}
	int getTail() {
		return this->tail;
	}

	Seg& operator=(Seg s) {
		this->head = s.head;
		this->tail = s.tail;
		this->width = s.width;
		return *this;
	}

private:
	int head;
	int tail;
	int width;
};


bool SegCompareByWidth(Seg a, Seg b);


bool SegCompareByHead(Seg a, Seg b);


// 根据（投影）向量生成 段向量
std::vector<Seg > vec2Segs(std::vector<int>  vec);


//去掉宽度远小于阈值的段
int segsFilter(std::vector<Seg>& segs, int threshold = 5);

//根据seg过滤对对应vec
int vecInSegPassFilter(std::vector<int>&vec, std::vector<Seg>segs);


//向量归一化
void vecNormalization(std::vector<double> &vec,int scale);

//向量的(N)范数
double vecNorm(std::vector<double> vec, double N);

//图像对比度增强
void contrastEnhancement(cv::Mat img);


//rect的排序
bool CompareRectByY(cv::Rect rect1, cv::Rect rect2);

bool CompareRectByX(cv::Rect rect1, cv::Rect rect2);
