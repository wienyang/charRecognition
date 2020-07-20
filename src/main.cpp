#include <opencv2/opencv.hpp>
#include <time.h>
#include <Windows.h>
#include <string>
#include <filesystem>
#include <fstream>
#include "findMrz.h"
#include "mrz.h"
#include "rotate.h"
#include "utils.h"
//#include "mrzAPI.h"


using namespace std;
using namespace cv;



#ifdef _DEBUG

namespace fs = filesystem;

//void segCharsTmp() {
//	string mrzDir = "F:/Project/mrzCharRec/data/mrz/20190513/";
//	string charDir = "F:/Project/mrzCharRec/data/gray/__20190513/";
//	vector<string> files;
//	getFiles(mrzDir, files, false);
//	for (auto file : files) {
//		string path = mrzDir + file;
//		MRZ mrz(path, 0);
//		mrz.segmentChars();
//		for (auto row : mrz.m_charImgMat) {
//			for (auto img : row) {
//				imwrite(charDir + randomStr() + ".jpg", img);
//			}
//		}
//	}
//}



int main(int argc, char* argv[])
{
	//if (argc < 4) {
	//	cout << "参数错误" << endl;
	//	return -1;
	//}
	//int type = atoi(argv[1]);
	//const char* imgDir = argv[2];
	//const char* modelPath = argv[3];

	//const char* io = argv[4];

	int type = 2;
	//const char* imgDir = "F:\\hahaha\\backup\\2020-01\\重要数据\\护照img\\type0";
	//const char* imgDir = "C:\\Users\\admin\\Desktop\\mrz32\\type2";
	const char* imgDir = "C:\\Users\\admin\\Desktop\\errorImg\\HX";
	const char* modelPath = "F:/hahaha/backup/2020-01/ocr/charRecognition/models/mrz32_61.pb";
	const char* io = "i";

	MRZ mrz(type);
	mrz.initNet(modelPath);
	cout << "初始化完成" << endl;

	for (auto& p : fs::directory_iterator(imgDir)) {
		if (p.path().extension() != ".bmp" && p.path().extension() != ".jpg" && p.path().extension() != ".png")
			continue;
		Mat img = imread(p.path().string(), 0);
		//namedWindow("test", 0);
		//imshow("test", img);

		TickMeter tm;
		tm.start();
		Mat mrzImg = findMrz(img, type);
		//Mat mrzImg = img;
		mrz.setImg(mrzImg);
		mrz.segmentChars();
		mrz.recognize();
		mrz.checkMrzStr();
		tm.stop();
		cout << "time:"<<tm.getTimeMilli() << endl;


		cout << mrz.mrzStr() << endl;
		/*if (strcmp(io, "i") == 0) {
			fs::path tmp = p.path();
			tmp.replace_extension(".txt");
			ifstream file(tmp.string());
			string label;
			while (!file.eof()) {
				string line;
				file >> line;
				label += line;
				label.push_back('\n');
			};
			label.pop_back();
			file.close();
			if (label == mrz.mrzStr())
				cout << "正确" << endl;
			else
				cout << "错误" << endl;
		}
		else if (strcmp(io, "o") == 0) {
			fs::path tmp = p.path();
			tmp.replace_extension(".txt");
			ofstream file(tmp.string());
			file << mrz.mrzStr();
			file.close();
		}*/
		waitKey();
		destroyAllWindows();
	}
	system("pause");
	return 0;
}



#endif


