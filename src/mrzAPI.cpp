
#include <opencv2/opencv.hpp>
#include <fstream>

#include "mrzAPI.h"
#include "mrz.h"
#include "findMrz.h"
#include "rotate.h"
#include "utils.h"

//#define GLOG_NO_ABBREVIATED_SEVERITIES 
//#include "glog/logging.h"




using namespace std;
using namespace cv;


bool glog_initialized = false;



int  mrzOcrAPI(const char* imgPath, int type, const char* modelPath, const char* outPath)
{
//#ifdef _DEBUG
//	FLAGS_log_dir = "../ljx";
//#else
//	FLAGS_log_dir = "./ljx";
//#endif
//	if (!glog_initialized)
//	{
//		google::InitGoogleLogging("mrzOcrAPI");
//		glog_initialized = true;
//	}
	MRZ mrz(imgPath, type);
	//LOG(INFO) << "MRZ construction done...";
	mrz.segmentChars();
	//LOG(INFO) << "segmentChars done...";
	mrz.initNet(modelPath);
	//LOG(INFO) << "initNet done...";
	mrz.recognize();
	//LOG(INFO) << "recognize done...";
	mrz.checkMrzStr();
	//LOG(INFO) << "checkMrzStr done...";
	string code = mrz.mrzStr();
	if (code.size() <= 0)
		code = "error!";

#ifdef PRINT
	cout << code << endl;
#endif
	ofstream out(outPath);
	out << code << endl;
	out.close();
	//LOG(INFO) << "result output done...";
	//if (glog_initialized)
	//{
		//google::ShutdownGoogleLogging();
		//glog_initialized = false;
	//}
	return 0;
}

int  findMrzAPI(const char* imgPath, int type, const char* outDir)
{
//#ifdef _DEBUG
//	FLAGS_log_dir = "../ljx";
//#else
//	FLAGS_log_dir = "./ljx";
//#endif
//	if (!glog_initialized)
//	{
//		google::InitGoogleLogging("findMrzAPI");
//		glog_initialized = true;
//	}
	string imgPathStr = imgPath;
	string outDtirStr = outDir;
	int ret = findMrz(imgPathStr, type, outDtirStr);
	//LOG(INFO) << "findMrz done...";
	/*if (glog_initialized)
	{
		google::ShutdownGoogleLogging();
		glog_initialized = false;
	}*/
	return ret;
}
