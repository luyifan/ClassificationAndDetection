#include <boost/python.hpp>
#include "Python.h"


#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <stdio.h>


#include <assert.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <math.h>
#include <time.h>
#include <fstream>
#include <strstream>
using namespace std;

#include <opencv2/opencv.hpp> 

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
using namespace cv;
#ifdef WIN32
/* windows stuff */
#else
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
typedef bool BOOL;
typedef void *HANDLE;
typedef unsigned char byte;
#endif
typedef vector<int> vecI;
typedef const string CStr;
typedef const Mat CMat;
typedef vector<string> vecS;
typedef vector<Mat> vecM;
typedef vector<float> vecF;
typedef vector<double> vecD;

enum{CV_FLIP_BOTH = -1, CV_FLIP_VERTICAL = 0, CV_FLIP_HORIZONTAL = 1};
#define _S(str) ((str).c_str())
#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
#define CV_Assert_(expr, args) \
{\
	if(!(expr)) {\
	string msg = cv::format args; \
	printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
	cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); }\
}

// Return -1 if not in the list
template<typename T>
static inline int findFromList(const T &word, const vector<T> &strList) {size_t idx = find(strList.begin(), strList.end(), word) - strList.begin(); return idx < strList.size() ? idx : -1;}
template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

inline Rect Vec4i2Rect(Vec4i &v){return Rect(Point(v[0] - 1, v[1] - 1), Point(v[2], v[3])); }
#ifdef __WIN32
    #define INT64 long long
#else
    #define INT64 long
    typedef unsigned long UINT64;
#endif

#define __POPCNT__
#include <immintrin.h>
#include <popcntintrin.h>
#ifdef __WIN32
# include <intrin.h>
# define POPCNT(x) __popcnt(x)
# define POPCNT64(x) __popcnt64(x)
#endif
#ifndef __WIN32
# define POPCNT(x) __builtin_popcount(x)
# define POPCNT64(x) __builtin_popcountll(x)
#endif




namespace bp = boost::python;

class Boxes{
public:
	Boxes(){ 
		value.clear();
		xmin.clear();
		xmax.clear();
		ymin.clear();
		ymax.clear();
	}
	void add(float v,int x1,int y1,int x2,int y2){
		value.push_back(v);
		xmin.push_back(x1);
		ymin.push_back(y1);
		xmax.push_back(x2);
		ymax.push_back(y2);
	}
	vector<float>::iterator valueBegin(){
		return value.begin();
	}
	vector<float>::iterator valueEnd(){
		return value.end();
	}
	vector<int>::iterator xminBegin(){
		return xmin.begin();
	}
	vector<int>::iterator xminEnd(){
		return xmin.end();
	}
	vector<int>::iterator yminBegin(){
		return ymin.begin();
	}
	vector<int>::iterator yminEnd(){
		return ymin.end();
	}
	vector<int>::iterator xmaxBegin(){
		return xmax.begin();
	}
	vector<int>::iterator xmaxEnd(){
		return xmax.end();
	}
	vector<int>::iterator ymaxBegin(){
		return ymax.begin();
	}
	vector<int>::iterator ymaxEnd(){
		return ymax.end();	
	}

private:
	vector<float> value;
	vector<int> xmin;
	vector<int> xmax;
	vector<int> ymin;
	vector<int> ymax;
};

class FilterTIG
{
public:
	void update(CMat &w1f){
		CV_Assert(w1f.cols * w1f.rows == D && w1f.type() == CV_32F && w1f.isContinuous());
		float b[D], residuals[D];
		memcpy(residuals, w1f.data, sizeof(float)*D);
		for (int i = 0; i < NUM_COMP; i++){
			float avg = 0;
			for (int j = 0; j < D; j++){
				b[j] = residuals[j] >= 0.0f ? 1.0f : -1.0f;
				avg += residuals[j] * b[j];
			}
			avg /= D;
			_coeffs1[i] = avg, _coeffs2[i] = avg*2, _coeffs4[i] = avg*4, _coeffs8[i] = avg*8;
			for (int j = 0; j < D; j++)
				residuals[j] -= avg*b[j];
			UINT64 tig = 0;
			for (int j = 0; j < D; j++)
				tig = (tig << 1) | (b[j] > 0 ? 1 : 0);
			_bTIGs[i] = tig;
		}
	}	

	// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
	Mat matchTemplate(const Mat &mag1u){
		const int H = mag1u.rows, W = mag1u.cols;
		const Size sz(W+1, H+1); // Expand original size to avoid dealing with boundary conditions
		Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);
		Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);
		Mat_<byte> Row1 = Mat_<byte>::zeros(sz), Row2 = Mat_<byte>::zeros(sz);
		Mat_<byte> Row4 = Mat_<byte>::zeros(sz), Row8 = Mat_<byte>::zeros(sz);
		Mat_<float> scores(sz);
		for(int y = 1; y <= H; y++){ 
			const byte* G = mag1u.ptr<byte>(y-1);
			INT64* T1 = Tig1.ptr<INT64>(y); // Binary TIG of current row
			INT64* T2 = Tig2.ptr<INT64>(y);
			INT64* T4 = Tig4.ptr<INT64>(y);
			INT64* T8 = Tig8.ptr<INT64>(y);
			INT64* Tu1 = Tig1.ptr<INT64>(y-1); // Binary TIG of upper row
			INT64* Tu2 = Tig2.ptr<INT64>(y-1);
			INT64* Tu4 = Tig4.ptr<INT64>(y-1);
			INT64* Tu8 = Tig8.ptr<INT64>(y-1);
			byte* R1 = Row1.ptr<byte>(y);
			byte* R2 = Row2.ptr<byte>(y);
			byte* R4 = Row4.ptr<byte>(y);
			byte* R8 = Row8.ptr<byte>(y);
			float *s = scores.ptr<float>(y);
			for (int x = 1; x <= W; x++) {
				byte g = G[x-1];
				R1[x] = (R1[x-1] << 1) | ((g >> 4) & 1);
				R2[x] = (R2[x-1] << 1) | ((g >> 5) & 1);
				R4[x] = (R4[x-1] << 1) | ((g >> 6) & 1);
				R8[x] = (R8[x-1] << 1) | ((g >> 7) & 1);
				T1[x] = (Tu1[x] << 8) | R1[x];
				T2[x] = (Tu2[x] << 8) | R2[x];
				T4[x] = (Tu4[x] << 8) | R4[x];
				T8[x] = (Tu8[x] << 8) | R8[x];
				s[x] = dot(T1[x], T2[x], T4[x], T8[x]);
			}
		}
		Mat matchCost1f;
		scores(Rect(8, 8, W-7, H-7)).copyTo(matchCost1f);
		return matchCost1f;
	}
	inline float dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8);

public:
	void reconstruct(Mat &w1f){// For illustration purpose
		w1f = Mat::zeros(8, 8, CV_32F);
		float *weight = (float*)w1f.data;
		for (int i = 0; i < NUM_COMP; i++){
			UINT64 tig = _bTIGs[i];
			for (int j = 0; j < D; j++)
				weight[j] += _coeffs1[i] * (((tig >> (63-j)) & 1) ? 1 : -1);
		}
	}
private:
	static const int NUM_COMP = 2; // Number of components
	static const int D = 64; // Dimension of TIG
	INT64 _bTIGs[NUM_COMP]; // Binary TIG features
	float _coeffs1[NUM_COMP]; // Coefficients of binary TIG features

	// For efficiently deals with different bits in CV_8U gradient map
	float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP]; 
};


inline float FilterTIG::dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8)
{
    INT64 bcT1 = __builtin_popcountll(tig1);
    INT64 bcT2 = __builtin_popcountll(tig2);
    INT64 bcT4 = __builtin_popcountll(tig4);
    INT64 bcT8 = __builtin_popcountll(tig8);
	
    INT64 bc01 = (__builtin_popcountll(_bTIGs[0] & tig1) << 1) - bcT1;
    INT64 bc02 = ((__builtin_popcountll(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
    INT64 bc04 = ((__builtin_popcountll(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
    INT64 bc08 = ((__builtin_popcountll(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

    INT64 bc11 = (__builtin_popcountll(_bTIGs[1] & tig1) << 1) - bcT1;
    INT64 bc12 = ((__builtin_popcountll(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
    INT64 bc14 = ((__builtin_popcountll(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
    INT64 bc18 = ((__builtin_popcountll(_bTIGs[1] & tig8) << 1) - bcT8) << 3;
    return _coeffs1[0] * (bc01 + bc02 + bc04 + bc08) + _coeffs1[1] * (bc11 + bc12 + bc14 + bc18);
}

template<typename VT, typename ST> 
struct ValStructVec
{
	ValStructVec(){clear();}
	inline int size() const {return sz;} 
	inline void clear() {sz = 0; structVals.clear(); valIdxes.clear();}
	inline void reserve(int resSz){clear(); structVals.reserve(resSz); valIdxes.reserve(resSz); }
	inline void pushBack(const VT& val, const ST& structVal) {valIdxes.push_back(make_pair(val, sz)); structVals.push_back(structVal); sz++;}
	inline const VT& operator ()(int i) const {return valIdxes[i].first;} // Should be called after sort
	inline const ST& operator [](int i) const {return structVals[valIdxes[i].second];} // Should be called after sort
	inline VT& operator ()(int i) {return valIdxes[i].first;} // Should be called after sort
	inline ST& operator [](int i) {return structVals[valIdxes[i].second];} // Should be called after sort

	void sort(bool descendOrder = true);
	const vector<ST> &getSortedStructVal();
	//void append(const ValStructVec<VT, ST> &newVals, int startV = 0);
	vector<ST> structVals; // struct values

private:
	int sz; // size of the value struct vector
	vector<pair<VT, int> > valIdxes; // Indexes after sort
	bool smaller() {return true;};
	vector<ST> sortedStructVals; 
};


template<typename VT, typename ST> 
void ValStructVec<VT, ST>::sort(bool descendOrder )
{
	if (descendOrder)
		std::sort(valIdxes.begin(), valIdxes.end(), std::greater<pair<VT, int> >());
	else
		std::sort(valIdxes.begin(), valIdxes.end(), std::less<pair<VT, int> >());
}

template<typename VT, typename ST> 
const vector<ST>& ValStructVec<VT, ST>::getSortedStructVal()
{
	sortedStructVals.resize(sz);
	for (int i = 0; i < sz; i++)
		sortedStructVals[i] = structVals[valIdxes[i].second];
	return sortedStructVals;
}

class Bing {
public:
	Bing(float base,int W,int NSS):_base(base),_W(W),_NSS(NSS),_logBase(log(_base))
				       ,_minT(cvCeil(log(10.)/_logBase))
				       ,_maxT(cvCeil(log(500.)/_logBase))
				       ,_numT(_maxT-_minT+1)
	{
	}
	int loadTrainModel(string modelName){
	   CStr s1 = modelName + ".wS1" , s2 = modelName + ".wS2" , sI = modelName + ".idx";
	   Mat filters1f, reW1f, idx1i, show3u;
	   if (!matRead(s1, filters1f) || !matRead(sI, idx1i)){
		   printf("Can't load model: %s or %s\n", _S(s1), _S(sI));		
		   return 0;
	   }
	   _tigF.update(filters1f);
	   _tigF.reconstruct(filters1f);
	   _svmSzIdxs = idx1i;
	   CV_Assert(_svmSzIdxs.size() > 1 && filters1f.size() == Size(_W, _W) && filters1f.type() == CV_32F);
	   _svmFilter = filters1f;
	   if (!matRead(s2, _svmReW1f) || _svmReW1f.size() != Size(2, _svmSzIdxs.size())){
		   _svmReW1f = Mat();
		   return -1;
	   }
	   return 1;
	}
	Boxes getBoxesOfOneImage(string imagefilename,int numDetPerSize,string storefilename){
		//cout << imagefilename << endl ;
		Mat img3u;
		ValStructVec<float,Vec4i> boxes;
		img3u = imread(_S(imagefilename));
		vecI sz;
		predictBBoxSI(img3u,boxes,sz,numDetPerSize,false);
		predictBBoxSII(boxes,sz);
		/*
		FILE *f = fopen(_S(storefilename), "w");
		fprintf(f, "%d\n", boxes.size());
		for (size_t k = 0; k < boxes.size(); k++)
			fprintf(f, "%g, %s\n", boxes(k), _S(strVec4i(boxes[k])));
		fclose(f);
		*/
		Boxes store_boxes;
		for (size_t k = 0; k < boxes.size(); k++){
			Vec4i box = boxes[k];
			store_boxes.add(boxes(k),box[0],box[1],box[2],box[3]);
		}
		return store_boxes;	
	}
	void predictBBoxSI(CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ, bool fast)
	{
		const int numSz = _svmSzIdxs.size();
		const int imgW = img3u.cols, imgH = img3u.rows;
		valBoxes.reserve(10000);
		sz.clear(); sz.reserve(10000);
		for (int ir = numSz - 1; ir >= 0; ir--){
			int r = _svmSzIdxs[ir];
			int height = cvRound(pow(_base, r/_numT + _minT)), width = cvRound(pow(_base, r%_numT + _minT));
			if (height > imgH * _base || width > imgW * _base)
				continue;
			height = min(height, imgH), width = min(width, imgW);
			Mat im3u, matchCost1f, mag1u;
			resize(img3u, im3u, Size(cvRound(_W*imgW*1.0/width), cvRound(_W*imgH*1.0/height)));
			gradientMag(im3u, mag1u);
			matchCost1f = _tigF.matchTemplate(mag1u);
			ValStructVec<float, Point> matchCost;
			nonMaxSup(matchCost1f, matchCost, _NSS, NUM_WIN_PSZ, fast);
			double ratioX = width/_W, ratioY = height/_W;
			int iMax = min(matchCost.size(), NUM_WIN_PSZ);
			for (int i = 0; i < iMax; i++){
				float mVal = matchCost(i);
				Point pnt = matchCost[i];
				Vec4i box(cvRound(pnt.x * ratioX), cvRound(pnt.y*ratioY));
				box[2] = cvRound(min(box[0] + width, imgW));
				box[3] = cvRound(min(box[1] + height, imgH));
				box[0] ++;
				box[1] ++;
				valBoxes.pushBack(mVal, box); 
				sz.push_back(ir);
			}
		}
	}

	void predictBBoxSII(ValStructVec<float, Vec4i> &valBoxes, const vecI &sz)
	{
		int numI = valBoxes.size();
		for (int i = 0; i < numI; i++){
			const float* svmIIw = _svmReW1f.ptr<float>(sz[i]);
			valBoxes(i) = valBoxes(i) * svmIIw[0] + svmIIw[1];
		}
		valBoxes.sort();
	}
	void nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS, int maxPoint, bool fast)
	{
		const int _h = matchCost1f.rows, _w = matchCost1f.cols;
		Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
		ValStructVec<float, Point> valPnt;
		matchCost.reserve(_h * _w);
		valPnt.reserve(_h * _w);
		if (fast){
			blur(matchCost1f, costSmooth1f, Size(3, 3));
			for (int r = 0; r < _h; r++){
				const float* d = matchCost1f.ptr<float>(r);
				const float* ds = costSmooth1f.ptr<float>(r);
				for (int c = 0; c < _w; c++)
					if (d[c] >= ds[c])
						valPnt.pushBack(d[c], Point(c, r));
			}
		}
		else{
			for (int r = 0; r < _h; r++){
				const float* d = matchCost1f.ptr<float>(r);
				for (int c = 0; c < _w; c++)
					valPnt.pushBack(d[c], Point(c, r));
			}
		}
		valPnt.sort();
		for (int i = 0; i < valPnt.size(); i++){
			Point &pnt = valPnt[i];
			if (isMax1u.at<byte>(pnt)){
				matchCost.pushBack(valPnt(i), pnt);
				for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
					Point neighbor = pnt + Point(dx, dy);
					if (!CHK_IND(neighbor))
						continue;
					isMax1u.at<byte>(neighbor) = false;
				}
			}
			if (matchCost.size() >= maxPoint)
				return;
		}
	}
	void gradientMag(CMat &imgBGR3u, Mat &mag1u){
		gradientRGB(imgBGR3u, mag1u); 
	}
	void gradientRGB(CMat &bgr3u, Mat &mag1u){
		const int H = bgr3u.rows, W = bgr3u.cols;
		Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);
		for (int y = 0; y < H; y++){
			Ix.at<int>(y, 0) = bgrMaxDist(bgr3u.at<Vec3b>(y, 1), bgr3u.at<Vec3b>(y, 0))*2;
			Ix.at<int>(y, W-1) = bgrMaxDist(bgr3u.at<Vec3b>(y, W-1), bgr3u.at<Vec3b>(y, W-2))*2;
		}
		for (int x = 0; x < W; x++){
			Iy.at<int>(0, x) = bgrMaxDist(bgr3u.at<Vec3b>(1, x), bgr3u.at<Vec3b>(0, x))*2;
			Iy.at<int>(H-1, x) = bgrMaxDist(bgr3u.at<Vec3b>(H-1, x), bgr3u.at<Vec3b>(H-2, x))*2;
		}
		for (int y = 0; y < H; y++){
			const Vec3b *dataP = bgr3u.ptr<Vec3b>(y);
			for (int x = 2; x < W; x++)
				Ix.at<int>(y, x-1) = bgrMaxDist(dataP[x-2], dataP[x]); //  bgr3u.at<Vec3b>(y, x+1), bgr3u.at<Vec3b>(y, x-1));
		}
		for (int y = 1; y < H-1; y++){
			const Vec3b *tP = bgr3u.ptr<Vec3b>(y-1);
			const Vec3b *bP = bgr3u.ptr<Vec3b>(y+1);
			for (int x = 0; x < W; x++)
				Iy.at<int>(y, x) = bgrMaxDist(tP[x], bP[x]);
		}
		gradientXY(Ix, Iy, mag1u);
	}
	void gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u){
		const int H = x1i.rows, W = x1i.cols;
		mag1u.create(H, W, CV_8U);
		for (int r = 0; r < H; r++){
			const int *x = x1i.ptr<int>(r), *y = y1i.ptr<int>(r);
			byte* m = mag1u.ptr<byte>(r);
			for (int c = 0; c < W; c++)
				m[c] = min(x[c] + y[c], 255);   //((int)sqrt(sqr(x[c]) + sqr(y[c])), 255);
		}
	}
	
	bool matRead(const string &filename,Mat& _M){
		FILE* f = fopen(_S(filename), "rb");
		if (f == NULL)
			return false;
		char buf[8];
		int pre = fread(buf,sizeof(char), 5, f);
		if (strncmp(buf, "CmMat", 5) != 0)	{
			printf("Invalidate CvMat data file %s\n", _S(filename));
			return false;
		}
		int headData[3]; 
		fread(headData, sizeof(int), 3, f);
		Mat M(headData[1], headData[0], headData[2]);
		fread(M.data, sizeof(char), M.step * M.rows, f);
		fclose(f);
		M.copyTo(_M);
		return true;
	}
private:
	float _base , _logBase;
	int _W;
	int _NSS;
	int _maxT,_minT,_numT;

	vecI _svmSzIdxs ;
	Mat _svmFilter;
	Mat _svmReW1f;
	FilterTIG _tigF;
	static inline int bgrMaxDist(const Vec3b &u, const Vec3b &v) {int b = abs(u[0]-v[0]), g = abs(u[1]-v[1]), r = abs(u[2]-v[2]); b = max(b,g);  return max(b,r);}
	inline string strVec4i(const Vec4i &v) const {return format("%d, %d, %d, %d", v[0], v[1], v[2], v[3]);}

};


BOOST_PYTHON_MODULE(bing){
	bp::class_<Bing>("Bing",bp::init<float,int,int>())
		.def("loadTrainModel",&Bing::loadTrainModel)
		.def("getBoxesOfOneImage",&Bing::getBoxesOfOneImage)
	;
	bp::class_<Boxes>("Boxes")
		.def("add",&Boxes::add)
		.def("values",bp::range(&Boxes::valueBegin,&Boxes::valueEnd))
		.def("xmins",bp::range(&Boxes::xminBegin,&Boxes::xminEnd))
		.def("xmaxs",bp::range(&Boxes::xmaxBegin,&Boxes::xmaxEnd))
		.def("ymins",bp::range(&Boxes::yminBegin,&Boxes::yminEnd))
		.def("ymaxs",bp::range(&Boxes::ymaxBegin,&Boxes::ymaxEnd))
	;
}



