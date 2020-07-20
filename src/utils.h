#pragma once
#include <vector>
#include <string>
#include <opencv2\opencv.hpp>


//��ȡ�������
std::vector<int> getDiffVec(std::vector<int> vec);

//��ȡĿ¼���ļ� 
void getFiles(std::string path, std::vector<std::string>& files, bool recurse);

//�����������
std::string randomStr();

//����Ӧ�ֲ���ֵ��
int adaptiveBinaryzation(cv::Mat &img);

//����ֵ��
int otsuBinaryzation(cv::Mat &img);

//HSV�ռ��ֵ�� 
int hsvBinaryzation(cv::Mat &img, int threshold);

//��ֵ��ͼ��ÿ�����ر仯����
int binaryImgHorizontalChange(cv::Mat &img, std::vector<int> &horizontalChangeVec);


//��ʾvector��Ӧֱ��ͼ
void showVecHist(std::vector<int> vec);

//��vector֮��

int vecSum(std::vector<int> vec);


//��vector�е�����ֵ��ֵ���ˣ���Ϊ0��

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


// ���ݣ�ͶӰ���������� ������
std::vector<Seg > vec2Segs(std::vector<int>  vec);


//ȥ�����ԶС����ֵ�Ķ�
int segsFilter(std::vector<Seg>& segs, int threshold = 5);

//����seg���˶Զ�Ӧvec
int vecInSegPassFilter(std::vector<int>&vec, std::vector<Seg>segs);


//������һ��
void vecNormalization(std::vector<double> &vec,int scale);

//������(N)����
double vecNorm(std::vector<double> vec, double N);

//ͼ��Աȶ���ǿ
void contrastEnhancement(cv::Mat img);


//rect������
bool CompareRectByY(cv::Rect rect1, cv::Rect rect2);

bool CompareRectByX(cv::Rect rect1, cv::Rect rect2);
