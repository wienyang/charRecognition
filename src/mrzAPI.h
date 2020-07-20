#pragma once
#include <string>



#ifdef MRZ_EXPORTS
#define MRZ_API __declspec(dllexport) 
#else
#define MRZ_API __declspec(dllimport) 
#endif

extern "C" {
	int MRZ_API mrzOcrAPI(const char* imgPath, int type, const char* modelPath, const char* outPath);
	int MRZ_API findMrzAPI(const char* imgPath, int type, const  char* outDir);
}