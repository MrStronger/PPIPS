#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
/*此类用于漫水算法演示*/

class Waterfloodfill
{
private:
    cv::Mat src, dst, gray, mask;//原图，结果图，灰度图，掩码图
	int g_nFillMode = 1;//漫水填充模式
	int g_nLowDifference = 20, g_nUpDifference = 20;//负差最大值，正差最大值
	int g_nConnectivity = 4;//表示floodFill函数标识符低八位的连通性
	int g_bIsColor = true;//是否为彩色图像的标识符布尔值
	bool g_bUseMask = false;//是否显示掩模窗口的布尔值
	int g_nNewMaskVal = 255;//新的重新绘制的像素值
	static void onMouse(int event, int x, int y, int Val, void* param);
public:
    Waterfloodfill(cv::Mat &image);
	void start();
	void setIsColor();
	void setmaskvisuable();
	void setFillMode(int type);//(0使用空范围的漫水填充)(1使用渐变，固定范围的漫水填充)（2使用渐变，浮动范围的漫水填充）
	void setConnectivity(int n);//（4操作标识符的第八位使用4位的连接模式）（8操作标识符的第八位使用8位的连接模式）
};
