#include "waterfloodfill.h"

Waterfloodfill::Waterfloodfill(cv::Mat & image)
{
	image.copyTo(src);
	image.copyTo(dst);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	mask.create(src.rows + 2, src.cols + 2, CV_8UC1);
}

void Waterfloodfill::start()
{
    cv::namedWindow("效果图", 1);
    cv::createTrackbar("负差最大值", "效果图", &g_nLowDifference, 255, 0);
    cv::createTrackbar("正差最大值", "效果图", &g_nUpDifference, 255, 0);
    cv::setMouseCallback("效果图", onMouse, this);
    cv::imshow("效果图", g_bIsColor ? dst : gray);
}

void Waterfloodfill::setIsColor()
{
	if (g_bIsColor)
	{
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        mask = cv::Scalar::all(0);
		g_bIsColor = false;
	}
	else
	{
		src.copyTo(dst);
        mask = cv::Scalar::all(0);
		g_bIsColor = true;
	}
}

void Waterfloodfill::setmaskvisuable()
{
	if (g_bUseMask)
	{
        cv::destroyWindow("mask");
		g_bUseMask = false;
	}
	else
	{
        cv::namedWindow("mask", 0);
        mask = cv::Scalar::all(0);
        cv::imshow("mask", mask);
		g_bUseMask = true;
	}
}

void Waterfloodfill::setFillMode(int type)
{
	g_nFillMode = type;
}

void Waterfloodfill::setConnectivity(int n)
{
	g_nConnectivity = n;
}

void Waterfloodfill::onMouse(int event, int x, int y, int Val, void * param)
{
	Waterfloodfill *p = (Waterfloodfill*)param;
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
    cv::Point seed = cv::Point(x, y);
	int LowDifference = p->g_nFillMode == 0 ? 0 : p->g_nLowDifference;
	int UpDifference = p->g_nFillMode == 0 ? 0 : p->g_nUpDifference;
	int flags = p->g_nConnectivity + (p->g_nNewMaskVal << 8) + (p->g_nFillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
    int b = (unsigned)cv::theRNG() & 255;
    int g = (unsigned)cv::theRNG() & 255;
    int r = (unsigned)cv::theRNG() & 255;
    cv::Rect ccomp;
    cv::Scalar newVal = p->g_bIsColor ? cv::Scalar(b, g, r) : cv::Scalar(r*0.299 + g*0.587 + b*0.114);
    cv::Mat g_dst = p->g_bIsColor ? p->dst : p->gray;
	if (p->g_bUseMask)
	{
        cv::threshold(p->mask, p->mask, 1, 128, cv::THRESH_BINARY);
        cv::floodFill(p->dst, p->mask, seed, newVal, &ccomp, cv::Scalar(LowDifference, LowDifference, LowDifference), cv::Scalar(UpDifference, UpDifference, UpDifference), flags);
        cv::imshow("mask", p->mask);
	}
	else
	{
        cv::floodFill(p->dst, seed, newVal, &ccomp, cv::Scalar(LowDifference, LowDifference, LowDifference), cv::Scalar(UpDifference, UpDifference, UpDifference), flags);
        cv::imshow("效果图", p->dst);
	}
}


