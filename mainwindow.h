#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include<QFileDialog>
#include<QMessageBox>
#include<QButtonGroup>
#include<string>

#include"ui_mainwindow.h"
#include"mainwindow1.h"
#include"clickablelabel.h"
#include "waterfloodfill.h"



//using namespace cv;



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    Ui::MainWindow *ui;
    MainWindow1* help;
    void openFile();




private slots://声明槽函数

void on_pushBtnopen_clicked();
void imageScaling(int value);
void imageTurn(int value);
void seePicture(cv::Mat pic);
QImage cvMat2QImage(const cv::Mat& mat);
cv::Mat QImage2cvMat(QImage *image);
void saveFile(QImage*);
void changePicture();


void on_toolButton_clicked();

void on_toolButton_2_clicked();

void on_toolButton_3_clicked();
void on_toolButton_4_clicked();

void on_toolButton_5_clicked();

void on_pushButton_7_clicked();

void on_pushBtnchange_clicked();

void on_pushButton_14_clicked();




int btnblur();//均值滤波
int btngaussian();//高斯滤波
int btnmedian();//中值滤波
int GetMedianNum(int *a);//冒泡排序
int btncaizhi();//彩色图像直方图

int btnsimple();//单通道直方图
int btnzhijun();//直方图均衡化
int btnCanny();//Canny边缘检测
void btnSobel(int value);//Sobel检测
void btnScharr();
int btnlight();//图像对比度与亮度
int btnzao(int value);//加噪
void ContrastAndBright(int value1,int value2);
void CannyThreshold(int value, void*);

int btnwave();//图像波纹化
int btncolorReduce(int value);//图像减色*/
int btnface();//人脸识别

void detectAndDisplay(cv::Mat frame);

void Relief();//浮雕处理
//凸透镜原理
void Expand();//扩张处理
void Squeezing();//挤压处理
void ColorMap();//使用自带的applycolormap进行颜色转换
void WaveSin();//正弦波浪效果


void on_toolButton_15_clicked();

void on_toolButton_16_clicked();

void on_toolButton_17_clicked();

void on_pushBtnover_clicked();

void on_toolButton_25_clicked();

void on_toolButton_20_clicked();

void on_toolButton_21_clicked();

void on_toolButton_22_clicked();

void on_toolButton_23_clicked();

void on_toolButton_24_clicked();

void on_toolButton_19_clicked();

void on_pushBtnmax_clicked();

void on_pushButton_31_clicked();

void on_toolButton_27_clicked();

void on_toolButton_28_clicked();

void on_toolButton_29_clicked();

void on_toolButton_6_clicked();

void on_toolButton_30_clicked();
void on_verticalSlider_5_valueChanged(int value);
void on_verticalSlider_valueChanged(int value);

void on_verticalSlider_2_valueChanged(int value);

void on_toolButton_18_clicked();

void on_toolButton_32_clicked();

void on_toolButton_33_clicked();

void on_verticalSlider_6_valueChanged(int value);

void on_dial_2_valueChanged(int value);

void on_dial_valueChanged(int value);

void on_verticalSlider_7_valueChanged(int value);





void on_toolButton_38_clicked();



void on_toolButton_40_clicked();

void on_toolButton_41_clicked();

void on_toolButton_42_clicked();

void on_toolButton_43_clicked();



void on_verticalSlider_3_valueChanged(int value);





void on_verticalSlider_4_valueChanged(int value);

void on_toolButton_7_clicked();

void on_toolButton_9_clicked();

void on_pushButton_26_clicked();





void on_toolButton_8_clicked();

void on_toolButton_10_clicked();

void on_toolButton_11_clicked();


void on_label_6_clicked();



void on_toolButton_26_clicked();

void on_toolButton_31_clicked();



void on_toolButton_14_clicked();

void on_toolButton_34_clicked();

void on_verticalSlider_9_valueChanged(int value);

void on_verticalSlider_8_valueChanged(int value);

void on_toolButton_12_clicked();

void on_open_clicked();

void on_pushButton_4_clicked();

void on_pushButton_5_clicked();

void on_pushButton_6_clicked();



void on_addAlpha_valueChanged(double arg1);

void on_addBeta_valueChanged(double arg1);

void on_pushButton_11_clicked();

void on_pushButton_15_clicked();

void on_toolButton_35_clicked();


protected:
    void paintEvent (QPaintEvent* );
    void mouseMoveEvent ( QMouseEvent * event );
    void mousePressEvent ( QMouseEvent * event );
    void mouseReleaseEvent(QMouseEvent *);
    void dragEnterEvent(QDragEnterEvent*event);
    void dropEvent(QDropEvent*event);

private:

    int silderstatus=1;//1.方块滤波2.均值滤波3.高斯滤波4.中值滤波5.双边滤波
    QPoint mousePosition;
    bool isMousePressed;
    int videostatus = 1;//视屏与摄像头状态
    QMatrix matrix;
    QImage* img=new QImage;
    QImage img2;
    QString path;
    cv::Mat img_box;
    cv::Mat  src,dst;
    cv::Mat  detected_edges;
    IplImage *histimg = 0;
    QButtonGroup *buttonGroup=new QButtonGroup;
    double alpha;
    int beta;
   /* cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    void detectAndDisplay(cv::Mat frame);*/

};

/*class WorkerThread : public QThread
{
    Q_OBJECT
    void run() Q_DECL_OVERRIDE {
        QString result;

        emit resultReady(result);
    }
};*/

#endif // MAINWINDOW_H
