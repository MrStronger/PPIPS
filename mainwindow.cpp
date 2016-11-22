#include "mainwindow.h"
#include "ui_mainwindow.h"



#include<QtGui>
#include<QDragEnterEvent>
#include<QtoolButton>
#include<QTextEdit>
#include<string>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    help=NULL;
    this->setWindowFlags(Qt::FramelessWindowHint);//去掉标题栏
    ui->label_38->hide();
    ui->label_41->hide();
    ui->verticalSlider_5->hide();
    ui->spinBox->hide();
    ui->radioButton->hide();
    ui->radioButton_2->hide();
    ui->radioButton_3->hide();
    ui->radioButton_4->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_10->hide();
    ui->label_25->hide();
    ui->pushButton->hide();
    ui->pushButton_11->hide();
    ui->pushButton_15->hide();
    ui->addAlpha->setEnabled(false);
    ui->addBeta->setEnabled(false);
    ui->verticalSlider_8->hide();
    ui->verticalSlider_9->hide();
    ui->label_9->hide();
    ui->spinBox_4->hide();


    ui->verticalSlider_7->hide();

    ui->spinBox_3->hide();

}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::mousePressEvent(QMouseEvent *event)
{
    mousePosition = event->pos();
//只对标题栏范围内的鼠标事件进行处理
    if (mousePosition.x()<=0)
        return;
    if ( mousePosition.x()>=1021)
        return;
    if (mousePosition.y()<=0 )
        return;
    if (mousePosition.y()>=101)
        return;
    isMousePressed = true;
}
void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if ( isMousePressed==true )
    {
        QPoint movePot = event->globalPos() - mousePosition;
        move(movePot);
    }
}
void MainWindow::mouseReleaseEvent(QMouseEvent *)
{
    isMousePressed=false;
}
 void MainWindow::paintEvent (QPaintEvent* )
 {
     QPainter painter(this);
     QPen pen(Qt::black);
     pen.setWidth(2);
     painter.setPen(pen);
     painter.drawRect(0,0,1022,642);
 }
 void MainWindow::dragEnterEvent(QDragEnterEvent *event)
 {
     if(event->mimeData()->hasUrls())
         event->acceptProposedAction();
     else event->ignore();
 }
 void MainWindow::dropEvent(QDropEvent *event)
 {
     const QMimeData*mimeData=event->mimeData();
     if(mimeData->hasUrls())
     {

         QList<QUrl>urlList=mimeData->urls();
         path=urlList.at(0).toLocalFile();

         src=cv::imread((const char *)path.toLocal8Bit());


         if(! ( img->load(path) ) ) //加载图像
         {
             QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


             delete img;
             return;
         }

         ui->label_3->setText(path);
         ui->label_6->setPixmap(QPixmap::fromImage(*img));
         ui->label_6->resize(img->width(), img->height());
         ui->scrollArea_2->setWidget(ui->label_6);
         ui->label_36->setPixmap(QPixmap::fromImage(*img));
         ui->label_36->resize(img->width(), img->height());
         ui->scrollArea_7->setWidget(ui->label_36);
         ui->label_37->setPixmap(QPixmap::fromImage(*img));
         ui->label_37->resize(img->width(), img->height());
         ui->scrollArea_12->setWidget(ui->label_37);



     }
 }
 void MainWindow::seePicture(cv::Mat pic)
 {

     if(ui->radioButton->isChecked()
             &&ui->radioButton_2->isChecked()
             &&ui->radioButton_3->isChecked()
             &&ui->radioButton_4->isChecked()
             &&ui->radioButton_5->isChecked())
     {
         cv::namedWindow("Preview");
         cv::imshow("Preview",src);
     }
     else{
     cv::namedWindow("Processed Image");
     cv::imshow("Processed Image",pic);
     }
 }
 void MainWindow::on_label_6_clicked()
 {
     seePicture(dst);
 }


 /****全局变量***/




    /*int erosion_elem = 0;             //腐蚀滚动条中滑块起始位置，也是返回值，代表核的不同形状
    int erosion_size = 0;   //腐蚀滚动条中滑块起始位置，也是返回值，代表核的大小
    int dilation_elem = 0;               //膨胀滚动条中滑块起始位置，也是返回值，代表核的不同形状
    int dilation_size = 0;           //膨胀滚动条中滑块起始位置，也是返回值，代表核的大小*/
    int elementshape=cv::MORPH_RECT;
    int const max_elem = 2;
    int const max_kernel_size = 21;

    int g_nContrastValue;//对比度
    int g_nBrightValue;//亮度值

    cv::Mat src_gray; 
    int edgeThresh = 1;
    int ThresholdType=0;
    int lowThreshold;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    double Alpha,Beta;
    const char* window_name = "Face recognition";
    cv::CascadeClassifier face_cascade;
       cv::CascadeClassifier eyes_cascade;


    int MainWindow::GetMedianNum(int *a)
    {
        int t;
        for (int j = 0; j < 9 - 1; j++) {
            for (int i = 0; i <= 9 - 1 - j; i++) {
                if (a[i] < a[i + 1]) {
                    t = a[i];
                    a[i] = a[i + 1];
                    a[i + 1] = t;
                }
            }
        }
        return a[5];
    }

/*IplImage* rotateImage(IplImage* src, int angle, bool clockwise)
{
    angle = abs(angle) % 180;
    if (angle > 90)
    {
        angle = 90 - (angle % 90);
    }
    IplImage* dst = NULL;
    int width =
        (double)(src->height * sin(angle * CV_PI / 180.0)) +
        (double)(src->width * cos(angle * CV_PI / 180.0)) + 1;
    int height =
        (double)(src->height * cos(angle * CV_PI / 180.0)) +
        (double)(src->width * sin(angle * CV_PI / 180.0)) + 1;
    int tempLength = sqrt((double)src->width * src->width + src->height * src->height) + 10;
    int tempX = (tempLength + 1) / 2 - src->width / 2;
    int tempY = (tempLength + 1) / 2 - src->height / 2;
    int flag = -1;

    dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);
    cvZero(dst);
    IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);
    cvZero(temp);

    cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height));
    cvCopy(src, temp, NULL);
    cvResetImageROI(temp);

    if (clockwise)
        flag = 1;

    float m[6];
    int w = temp->width;
    int h = temp->height;
    m[0] = (float)cos(flag * angle * CV_PI / 180.);
    m[1] = (float)sin(flag * angle * CV_PI / 180.);
    m[3] = -m[1];
    m[4] = m[0];
    // 将旋转中心移至图像中间
    m[2] = w * 0.5f;
    m[5] = h * 0.5f;
    //
    CvMat M = cvMat(2, 3, CV_32F, m);
    cvGetQuadrangleSubPix(temp, dst, &M);
    cvReleaseImage(&temp);
    return dst;
}*/


void MainWindow::detectAndDisplay(cv::Mat frame)
{

    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    //多尺寸检测人脸
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for (int i = 0; i < faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        ellipse(frame, center, cv::Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;
        //检测人眼
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        for (int j = 0; j < eyes.size(); j++)
        {
            cv::Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
        }
    }
    cv::namedWindow(window_name);
    imshow(window_name, frame);

}

/***实现Mat图像文件与QImage图像的转换***/
QImage cvMat2QImage(const cv::Mat& mat)
{
        // 8-bits unsigned, NO. OF CHANNELS = 1
        if(mat.type() == CV_8UC1)
        {
            QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
            // Set the color table (used to translate colour indexes to qRgb values)
            image.setColorCount(256);
            for(int i = 0; i < 256; i++)
            {
                image.setColor(i, qRgb(i, i, i));
            }
            // Copy input Mat
            uchar *pSrc = mat.data;
            for(int row = 0; row < mat.rows; row ++)
            {
                uchar *pDest = image.scanLine(row);
                memcpy(pDest, pSrc, mat.cols);
                pSrc += mat.step;
            }
            return image;
        }
        // 8-bits unsigned, NO. OF CHANNELS = 3
        else if(mat.type() == CV_8UC3)
        {
            // Copy input Mat
            const uchar *pSrc = (const uchar*)mat.data;
            // Create QImage with same dimensions as input Mat
            QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
            return image.rgbSwapped();
        }
        else if(mat.type() == CV_8UC4)
        {
            qDebug() << "CV_8UC4";
            // Copy input Mat
            const uchar *pSrc = (const uchar*)mat.data;
            // Create QImage with same dimensions as input Mat
            QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
            return image.copy();
        }
        else
        {
            qDebug() << "ERROR: Mat could not be converted to QImage.";
            return QImage();
        }
    }

QImage MainWindow::cvMat2QImage(const cv::Mat& mat)
{
        // 8-bits unsigned, NO. OF CHANNELS = 1
        if(mat.type() == CV_8UC1)
        {
            QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
            // Set the color table (used to translate colour indexes to qRgb values)
            image.setColorCount(256);
            for(int i = 0; i < 256; i++)
            {
                image.setColor(i, qRgb(i, i, i));
            }
            // Copy input Mat
            uchar *pSrc = mat.data;
            for(int row = 0; row < mat.rows; row ++)
            {
                uchar *pDest = image.scanLine(row);
                memcpy(pDest, pSrc, mat.cols);
                pSrc += mat.step;
            }
            return image;
        }
        // 8-bits unsigned, NO. OF CHANNELS = 3
        else if(mat.type() == CV_8UC3)
        {
            // Copy input Mat
            const uchar *pSrc = (const uchar*)mat.data;
            // Create QImage with same dimensions as input Mat
            QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
            return image.rgbSwapped();
        }
        else if(mat.type() == CV_8UC4)
        {
            qDebug() << "CV_8UC4";
            // Copy input Mat
            const uchar *pSrc = (const uchar*)mat.data;
            // Create QImage with same dimensions as input Mat
            QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
            return image.copy();
        }
        else
        {
            qDebug() << "ERROR: Mat could not be converted to QImage.";
            return QImage();
        }
    }


void MainWindow::CannyThreshold(int value, void*)
{
    /// 使用 3x3内核降噪
    cv::blur(src_gray, detected_edges, cv::Size(3, 3));

    /// 运行Canny算子，支持原地计算。根据Canny算法的推荐，Canny阈值输入比例1:3，核大小为3*3
    cv::Canny(detected_edges, detected_edges, value, value*ratio, kernel_size);

    /// 黑色背景
    //dst = cv::Scalar::all(0);

    // copyTo 将 detected_edges 拷贝到 dst . 但是，仅仅拷贝掩码不为0的像素。
    //Canny边缘检测的输出是镶嵌在黑色背景中的边缘像素，因此其结果 dst 图像除了被检测的边缘像素，其余部分都为黑色。
    src.copyTo(dst, detected_edges);

    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(detected_edges)));
    //imshow(window_name, dst);
    //imshow( window_name, detected_edges );
}

void MainWindow::openFile()//打开文件

{
    if(path!=":/images/exm1.jpg"&&path!=":/images/exm1.jpg"&&path!=":/images/exm1.jpg")
    path = QFileDialog::getOpenFileName(this,tr("打开文件"), "/", tr("Images (*.png *.bmp *.jpg *.tif *.GIF *.mp4 *.MPG)"));

    if(!path.isEmpty())
    {
                src=cv::imread((const char *)path.toLocal8Bit());


                if( img->load(path) ) //加载图像
                {

                    ui->frame->hide();

                    ui->label_3->setText(path);
                    ui->label_6->setPixmap(QPixmap::fromImage(*img));
                    ui->label_6->resize(img->width(), img->height());
                    ui->scrollArea_2->setWidget(ui->label_6);

                    ui->label_36->setPixmap(QPixmap::fromImage(*img));
                    ui->label_36->resize(img->width(), img->height());
                    ui->scrollArea_7->setWidget(ui->label_36);
                    ui->label_37->setPixmap(QPixmap::fromImage(*img));
                    ui->label_37->resize(img->width(), img->height());
                    ui->scrollArea_12->setWidget(ui->label_37);
                }else{
                    QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


                    //delete img;
                    return;
                }




     }
     else
    {
        QMessageBox::warning(this, tr("Path"),tr("You did not select any file."));
    }

}
void MainWindow::saveFile(QImage* img2)//保存文件
{
    path = QFileDialog::getSaveFileName(this,tr("保存图片"), path,tr("Images (*.png *.bmp *.jpg *.tif *.GIF )")); //选择路径



    if(path.isEmpty())
    {
        QMessageBox::warning(this, tr("Path"),tr("You did not select any file."));
    }
    else
    {
        if(! ( img2->save(path) ) ) //保存图像
        {
            QMessageBox::information(this, tr("Failed to save the image"), tr("Failed to save the image!"));


            return;
        }
    }
}
/**QImage变成Mat**/
cv::Mat QImage2cvMat(QImage *image)
{
    cv::Mat mat;
    qDebug() << image->format();
    switch(image->format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image->height(), image->width(), CV_8UC4, (void*)image->constBits(), image->bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image->height(), image->width(), CV_8UC3, (void*)image->constBits(), image->bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);

        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image->height(), image->width(), CV_8UC1, (void*)image->constBits(), image->bytesPerLine());

        break;
    }
    cv::cvtColor(mat,mat,CV_BGRA2BGR);
    return mat;
}



cv::Mat MainWindow::QImage2cvMat(QImage *image)
{
    cv::Mat mat;
    qDebug() << image->format();
    switch(image->format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image->height(), image->width(), CV_8UC4, (void*)image->constBits(), image->bytesPerLine());

        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image->height(), image->width(), CV_8UC3, (void*)image->constBits(), image->bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);

        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image->height(), image->width(), CV_8UC1, (void*)image->constBits(), image->bytesPerLine());

        break;
    }
    cv::cvtColor(mat,mat,CV_BGRA2BGR);
    return mat;
}
/***IplImage转换为QImage***/
IplImage *QImageToIplImage(const QImage * qImage)
{
    int width = qImage->width();
    int height = qImage->height();

    CvSize Size;
    Size.height = height;
    Size.width = width;

    IplImage *charIplImageBuffer = cvCreateImage(Size, IPL_DEPTH_8U, 1);
    char *charTemp = (char *) charIplImageBuffer->imageData;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = y * width + x;
            charTemp[index] = (char) qGray(qImage->pixel(x, y));
        }
    }

    return charIplImageBuffer;
}

QImage IplImgToQImg(IplImage* cvimage)
{
    QImage destImage(cvimage->width,cvimage->height,QImage::Format_RGB32);
    for(int i = 0; i < cvimage->height; i++)
    {
        for(int j = 0; j < cvimage->width; j++)
        {
            int r,g,b;
            if(3 == cvimage->nChannels)
            {
                b=(int)CV_IMAGE_ELEM(cvimage,uchar,i,j*3+0);
                g=(int)CV_IMAGE_ELEM(cvimage,uchar,i,j*3+1);
                r=(int)CV_IMAGE_ELEM(cvimage,uchar,i,j*3+2);
            }
            else if(1 == cvimage->nChannels)
            {
                b=(int)CV_IMAGE_ELEM(cvimage,uchar,i,j);
                g=b;
                r=b;
            }
                destImage.setPixel(j,i,qRgb(r,g,b));
         }
    }
    return destImage;
}


void MainWindow::on_pushBtnopen_clicked()
{
    if(ui->pushButton_4->isChecked())ui->pushButton_4->setChecked(false);
    else if(ui->pushButton_5->isChecked())ui->pushButton_5->setChecked(false);
    else if(ui->pushButton_6->isChecked())ui->pushButton_6->setChecked(false);
    openFile();
}


/***图像缩放****/
void MainWindow::imageScaling(int value)
{   img2=img->scaled(620*value/10,412*value/10,Qt::KeepAspectRatio);
    ui->label_6->setPixmap(QPixmap::fromImage(img2));
}

/***图像旋转***/
void MainWindow::imageTurn(int value)
{

    img2=img->transformed(matrix);
    matrix.rotate(36*value);

    ui->label_6->setPixmap(QPixmap::fromImage(img2));

}







void MainWindow::on_toolButton_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(0);
    ui->stackedWidget_4->setCurrentIndex(0);
    ui->stackedWidget_5->setCurrentIndex(0);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);

}

void MainWindow::on_toolButton_2_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(2);
    ui->stackedWidget_4->setCurrentIndex(2);
    ui->stackedWidget_5->setCurrentIndex(0);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);
}

void MainWindow::on_toolButton_3_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(3);
    ui->stackedWidget_4->setCurrentIndex(3);
    ui->stackedWidget_5->setCurrentIndex(0);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);

}

void MainWindow::on_toolButton_4_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(4);
    ui->stackedWidget_4->setCurrentIndex(4);
    ui->stackedWidget_5->setCurrentIndex(0);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);
}

void MainWindow::on_toolButton_5_clicked()
{
    ui->toolButton_19->hide();
    ui->pushBtnchange->hide();
    ui->pushButton_7->hide();
    ui->stackedWidget_2->setCurrentIndex(6);
    ui->stackedWidget_4->setCurrentIndex(6);
    ui->stackedWidget_5->setCurrentIndex(2);
    ui->pushBtnopen->setEnabled(false);
    ui->pushButton_14->setEnabled(false);
    ui->radioButton->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_2->setChecked(true);
}



void MainWindow::on_pushButton_7_clicked()
{
    if(ui->toolButton_27->isChecked())saveFile(&cvMat2QImage(img_box));
    else if(ui->toolButton_28->isChecked())saveFile(&IplImgToQImg(histimg));
    if(ui->toolButton_15->isChecked())saveFile(&cvMat2QImage(detected_edges));
    else if(ui->toolButton_30->isChecked())saveFile(&img2);
    else saveFile(&cvMat2QImage(dst));

    //saveFile(&cvMat2QImage(dst));
}

void MainWindow::ContrastAndBright(int value1, int value2)
{
    //创建窗口
    //cv::namedWindow("img", 1);

    //三个for函数，执行计算
    for (int y = 0; y <src.rows; y++)
    {
        for (int x = 0; x <src.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                dst.at<cv::Vec3b>(y, x)[c] =cv::saturate_cast<uchar>((value2*0.01)*(src.at<cv::Vec3b>(y, x)[c]) + value1);
            }
        }
    }
    //显示图像
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    //imshow("img", src);
    //imshow("img2", dst);
}

void MainWindow::changePicture()
{

    path = QFileDialog::getOpenFileName(this,tr("换张图片"), path, tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));

    if(!path.isEmpty())
    {

                src=cv::imread((const char *)path.toLocal8Bit());

                if(! ( img->load(path) ) ) //加载图像
                {
                    QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


                    delete img;
                    return;
                }

                ui->label_3->setText(path);
                ui->label_6->setPixmap(QPixmap::fromImage(*img));
                ui->label_6->resize(img->width(), img->height());
                ui->scrollArea_2->setWidget(ui->label_6);
                ui->label_36->setPixmap(QPixmap::fromImage(*img));
                ui->label_36->resize(img->width(), img->height());
                ui->scrollArea_7->setWidget(ui->label_36);
                ui->label_37->setPixmap(QPixmap::fromImage(*img));




     }
     else
    {
        QMessageBox::warning(this, tr("Tips"),tr("You did not select any file."));
    }
}

void MainWindow::on_pushBtnchange_clicked()
{
    if(ui->pushButton_4->isChecked())ui->pushButton_4->setChecked(false);
    else if(ui->pushButton_5->isChecked())ui->pushButton_5->setChecked(false);
    else if(ui->pushButton_6->isChecked())ui->pushButton_6->setChecked(false);
    changePicture();
    if(src.empty())ui->radioButton->setChecked(true);

}







int MainWindow::btnblur()
{
    src.copyTo(dst);
    if (src.channels() == 3) {
        for (int y = 1; y < src.rows - 1; y++)
        {
            uchar *previous = src.ptr<uchar>(y - 1);
            uchar *current = src.ptr<uchar>(y);
            uchar *next = src.ptr<uchar>(y + 1);

            uchar *output = dst.ptr<uchar>(y);
            if (ui->pushButton->isChecked()) break;
            for (int x = 1; x < src.cols - 1; x++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    output[3 * x + c] = cv::saturate_cast<int>((previous[3 * x + c - 3] + previous[3 * x + c] + previous[3 * x + c + 3] + current[3 * x + c - 3] + current[3 * x + c] + current[3 * x + c + 3] + next[3 * x + c - 3] + next[3 * x + c] + next[3 * x + c + 3]) / 9);
                }
                img_box = dst.clone();
                cv::rectangle(img_box, cv::Rect(x + 1, y, 9, 9), cv::Scalar(0, 0, 255));
                Sleep(10);//等待一毫秒
                cv::namedWindow("blur",cv::WINDOW_NORMAL);
                cv::imshow("blur",img_box);

                if (cv::waitKey(1) == 27||ui->pushButton->isChecked()) break;
            }
        }
    }


    return 0;
}//均值滤波
int MainWindow::btngaussian()
{
    src.copyTo(dst);
    if (src.channels() == 3) {
        for (int y = 1; y < src.rows - 1; y++)
        {
            uchar *previous = src.ptr<uchar>(y - 1);
            uchar *current = src.ptr<uchar>(y);
            uchar *next = src.ptr<uchar>(y + 1);

            uchar *output = dst.ptr<uchar>(y);
            if (ui->pushButton->isChecked()) break;
            for (int x = 1; x < src.cols - 1; x++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    output[3 * x + c] = cv::saturate_cast<int>((previous[3 * x + c - 3] + previous[3 * x + c] * 2 + previous[3 * x + c + 3] + current[3 * x + c - 3] * 2 + current[3 * x + c] * 4 + current[3 * x + c + 3] * 2 + next[3 * x + c - 3] + next[3 * x + c] * 2 + next[3 * x + c + 3]) / 16);
                }
                img_box = dst.clone();
                cv::rectangle(img_box, cv::Rect(x + 1, y, 9, 9), cv::Scalar(0, 0, 255));
                Sleep(10);//等待一毫秒
                cv::namedWindow("gaussian",cv::WINDOW_NORMAL);
                cv::imshow("gaussian",img_box);
                if (cv::waitKey(1) == 27||ui->pushButton->isChecked()) break;
            }
        }
    }
    return 0;
}//高斯滤波
int MainWindow::btnmedian()
{
    src.copyTo(dst);
    if (src.channels() == 3) {
        for (int y = 1; y < src.rows - 1; y++)
        {
            uchar *previous = src.ptr<uchar>(y - 1);
            uchar *current = src.ptr<uchar>(y);
            uchar *next = src.ptr<uchar>(y + 1);

            uchar *output = dst.ptr<uchar>(y);
            if (ui->pushButton->isChecked()) break;
            for (int x = 1; x < src.cols - 1; x++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    int temparr[9];
                    temparr[0] = current[3 * x + c];
                    temparr[1] = current[3 * x + c - 3];
                    temparr[2] = current[3 * x + c + 3];
                    temparr[3] = previous[3 * x + c];
                    temparr[4] = previous[3 * x + c - 3];
                    temparr[5] = previous[3 * x + c + 3];
                    temparr[6] = next[3 * x + c];
                    temparr[7] = next[3 * x + c - 3];
                    temparr[8] = next[3 * x + c + 3];
                    output[3 * x + c] = GetMedianNum(temparr);
                 }
                img_box = dst.clone();
                cv::rectangle(img_box, cv::Rect(x + 1, y, 9, 9), cv::Scalar(0, 0, 255));
                Sleep(10);//等待一毫秒
                cv::namedWindow("median",cv::WINDOW_NORMAL);
                cv::imshow("median",img_box);
                if (cv::waitKey(1) == 27||ui->pushButton->isChecked()) break;
            }
        }
    }
    return 0;
}
//中值滤波

void MainWindow::on_pushButton_14_clicked()
{
    if(ui->stackedWidget_4->currentIndex()==0&&ui->radioButton->isChecked()
            ||ui->stackedWidget_4->currentIndex()==3&&ui->radioButton_2->isChecked()
            ||(ui->stackedWidget_4->currentIndex()==2)&&ui->radioButton_3->isChecked()
            ||ui->stackedWidget_4->currentIndex()==4&&ui->radioButton_4->isChecked()
            ||ui->stackedWidget_4->currentIndex()==5&&ui->radioButton_5->isChecked()
            ||ui->stackedWidget_4->currentIndex()==1&&ui->radioButton_6->isChecked())

    {QMessageBox message(QMessageBox::Information, "提示", "您需要先选择功能才能开始处理！", QMessageBox::Yes, this);
        message.exec(); }
    else if(ui->pushButton->isChecked())ui->pushButton->setChecked(false);
    else if(ui->toolButton_27->isChecked())btncaizhi();
    else if(ui->toolButton_28->isChecked())btnsimple();
    else if(ui->toolButton_29->isChecked())btnzhijun();
    else if(ui->toolButton_20->isChecked())btnblur();
    else if(ui->toolButton_10->isChecked())btngaussian();
    else if(ui->toolButton_11->isChecked())btnmedian();
    else if(ui->toolButton_32->isChecked())btnwave();
    else if(ui->toolButton_7->isChecked())btnScharr();
    else if(ui->toolButton_21->isChecked()){
        cv::boxFilter(src,dst, -1, cv::Size(25 + 1,25 + 1));
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));}
    else if(ui->toolButton_22->isChecked()){
        cv::boxFilter(src,dst, -1, cv::Size(25 + 1,25 + 1));
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if(ui->toolButton_23->isChecked()){
        cv::GaussianBlur(src,dst,cv::Size(25*2 + 1,25*2 + 1),0,0);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if(ui->toolButton_24->isChecked()){
        cv::medianBlur(src,dst,25*2+1 );
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if(ui->toolButton_25->isChecked()){
        cv::bilateralFilter(src,dst,25,25*2,25/2);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if(ui->toolButton_38->isChecked())ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));

    else if(ui->toolButton_40->isChecked())ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    else if(ui->toolButton_41->isChecked())ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    else if(ui->toolButton_42->isChecked())ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    else if(ui->toolButton_43->isChecked())ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));

    else ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(img_box)));
}
int MainWindow::btncaizhi()//彩色图像直方图
{


    //装载图像


    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_3->setChecked(true);
    }
    else{
    // 分割成3个单通道图像 ( R, G 和 B )
    std::vector<cv::Mat> rgb_planes;
    cv::split(src, rgb_planes);

    // 设定bin数目
    int histSize = 256;

    // 设定取值范围 ( R,G,B) )
    float range[] = { 0, 255 };       //上下界区间
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    //储存直方图的矩阵
    cv::Mat r_hist, g_hist, b_hist;

    // 计算直方图:
    //&rgb_planes[0]: 输入数组(或数组集)
    //1: 输入数组的个数 (这里我们使用了一个单通道图像，我们也可以输入数组集 )
    //0: 需要统计的通道 (dim)索引 ，这里我们只是统计了灰度 (且每个数组都是单通道)所以只要写 0 就行了。
    //Mat(): 掩码( 0 表示忽略该像素)， 如果未定义，则不使用掩码
    //r_hist: 储存直方图的矩阵
    //1: 直方图维数
    //histSize: 每个维度的bin数目
    //histRange: 每个维度的取值范围
    //uniform 和 accumulate: bin大小相同，清楚直方图痕迹
    cv::calcHist(&rgb_planes[0], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&rgb_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&rgb_planes[2], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

    // 创建直方图画布
    int hist_h = 400;
    int hist_w = 256 * 3;
    //int bin_w = cvRound( (double) 256/histSize );             //bin的宽度

    img_box=cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));  //行数400，列数1200的直方图图像

                                                              // 将直方图归一化到范围 [ 0, histImage.rows ]，这样画直方图的时候不会超出图片的高度，高度400
    cv::normalize(r_hist, r_hist, 0, img_box.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, img_box.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(b_hist, b_hist, 0, img_box.rows, cv::NORM_MINMAX, -1, cv::Mat());


    for (int i = 1; i < histSize; i++)
    {
        cv::rectangle(img_box, cv::Point(i - 1, hist_h - 1), cv::Point(i, hist_h - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255), 1, 8, 0);
        cv::rectangle(img_box, cv::Point(i - 1 + 256, hist_h - 1), cv::Point(i + 256, hist_h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0), 1, 8, 0);
        cv::rectangle(img_box, cv::Point(i - 1 + 512, hist_h - 1), cv::Point(i + 512, hist_h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 1, 8, 0);

    }
    //显示直方图
   /* cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
    imshow("calcHist Demo", img_box);
    cv::waitKey(0);*/
    ui->label_37->setPixmap(QPixmap::fromImage(cvMat2QImage(img_box)));
    ui->label_37->resize(cvMat2QImage(img_box).width(), cvMat2QImage(img_box).height());
    ui->scrollArea_12->setWidget(ui->label_37);

    return 0;
    }


}

int MainWindow::btnsimple()//单通道直方图
{
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_6->setChecked(true);
    }
    else{
    IplImage *src = 0;

    CvHistogram *hist = 0;                    //直方图

                                              //  int hdims = 50;     // 划分HIST的个数，越高越精确
                                              //  int hdims=100;
    int hdims = 255;
    float hranges_arr[] = { 0,255 };
    float* hranges = hranges_arr;
    int bin_w;
    float max_val;
    int i;

    if(ui->pushButton_4->isChecked())path="exm1.jpg";
    else if(ui->pushButton_5->isChecked())path="exm2.jpg";
    else if(ui->pushButton_6->isChecked())path="exm3.jpg";

    src = cvLoadImage((const char *)path.toLocal8Bit(), 0);//灰度读入

    //cvNamedWindow("Histogram", 0);
    //cvNamedWindow("src", 0);

    hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);   //创建直方图
    histimg = cvCreateImage(cvSize(320, 200), 8, 3);
    cvZero(histimg);                                            //清零
    cvCalcHist(&src, hist, 0, 0);                               // 计算直方图,即每个bin的大小
    cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0);              // 只找最大值
    cvConvertScale(hist->bins,
        hist->bins, max_val ? 255. / max_val : 0., 0);             // 缩放 bin 到区间 [0,255]
    cvZero(histimg);
    bin_w = histimg->width / hdims;                               // hdims: 直方图竖条的个数，则 bin_w 为条的宽度

                                                                  // 画直方图
    for (i = 0; i < hdims; i++)
    {
        double val = (cvGetReal1D(hist->bins, i)*histimg->height / 255);
        CvScalar color = CV_RGB(255, 255, 0);                                 //(hsv2rgb(i*180.f/hdims);
        cvRectangle(histimg, cvPoint(i*bin_w, histimg->height),
            cvPoint((i + 1)*bin_w, (int)(histimg->height - val)),
            color, 1, 8, 0);
    }

    //cvShowImage("src", src);
    //cvShowImage("Histogram", histimg);
    ui->label_36->setPixmap(QPixmap::fromImage(IplImgToQImg(src)));
    ui->label_36->resize(img->width(), img->height());
    ui->scrollArea_7->setWidget(ui->label_36);
    ui->label_37->setPixmap(QPixmap::fromImage(IplImgToQImg(histimg)));
    ui->label_37->resize(img->width(), img->height());
    ui->scrollArea_12->setWidget(ui->label_37);
    cvWaitKey(0);
    //cvDestroyWindow("src");
    //vDestroyWindow("Histogram");
    cvReleaseImage(&src);
    cvReleaseImage(&histimg);
    cvReleaseHist(&hist);

    return 0;
    }
}

int MainWindow::btnzhijun()//直方图均衡化
{


    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_6->setChecked(true);
    }
    else{
    // 转为灰度图,原地转换
    cvtColor(src, src, CV_BGR2GRAY);

    // 应用直方图均衡化
    equalizeHist(src, dst);

    // 显示结果
    //cv::namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    //cv::namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
    ui->label_36->setPixmap(QPixmap::fromImage(cvMat2QImage(src)));
    ui->label_36->resize(img->width(), img->height());
    ui->scrollArea_7->setWidget(ui->label_36);
    ui->label_37->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    ui->label_37->resize(img->width(), img->height());
    ui->scrollArea_12->setWidget(ui->label_37);
    //imshow(source_window, src);
    //imshow(equalized_window, dst);

    // 等待用户按键退出程序
    cv::waitKey(0);
    return 0;
    }
}


/*void MainWindow::choise(int index)
{
   switch (index)
    {
     case 0:{disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnzhijun()));disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnsimple()));connect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btncaizhi()));}break;
     case 1:{disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnzhijun()));disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btncaizhi()));connect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnsimple()));}break;
     case 2:{disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnsimple()));disconnect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btncaizhi()));connect(ui->toolButton_14,SIGNAL(clicked(bool)),this,SLOT(btnzhijun()));}break;

     default:break;
    }

}*/

int MainWindow::btnCanny()//Canny边缘检测
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else{
    /// 创建与src同类型和大小的矩阵(dst)
    dst.create(src.size(), src.type());

    /// 原图像转换为灰度图像
    cvtColor(src, src_gray, CV_BGR2GRAY);
    //cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);

    /// 创建trackbar
    //cv::createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold,CannyThreshold);

    /// 显示图像，用于第一次没有移动滑块的时候
    CannyThreshold(0, 0);

    cv::waitKey(0);
    return 0;
    }
}
void MainWindow::btnSobel(int value)
{
    cv::Mat g_sobel_x,g_sobel_y;


    cv::Sobel(src, g_sobel_x, CV_16S, 1, 0, (2 * value + 1), 1, 1);
    cv::convertScaleAbs(g_sobel_x, g_sobel_x);
    cv::Sobel(src, g_sobel_y, CV_16S, 0, 1, (2 * value + 1), 1, 1);
    cv::convertScaleAbs(g_sobel_y, g_sobel_y);
    cv::addWeighted(g_sobel_x, 0.5, g_sobel_y, 0.5, 0, dst);
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
}

void MainWindow::btnScharr()
{
    cv::Mat g_scharr_x,g_scharr_y;
    cv::Scharr(src, g_scharr_x, CV_16S, 1, 0, 1, 0);
    cv::convertScaleAbs(g_scharr_x, g_scharr_x);
    cv::Scharr(src, g_scharr_y, CV_16S, 0, 1, 1, 0);
    cv::convertScaleAbs(g_scharr_y, g_scharr_y);
    cv::addWeighted(g_scharr_x, 0.5, g_scharr_y, 0.5, 0, dst);
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
}



int MainWindow::btnlight()//图像对比度与亮度
{



    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_3->setChecked(true);

    }
    else{
    dst = cv::Mat::zeros(src.size(), src.type());

    //设定对比度和亮度初始值
    //g_nContrastValue = 80;
    //g_nBrightValue = 80;

    //创建窗口
    //cv::namedWindow("img", 1);

    //创建轨迹条
    //cv::createTrackbar("cont", "img2", &value1, 300, ContrastAndBright);
    //cv::createTrackbar("ligh", "img2", &value2, 200, ContrastAndBright);

    //调用回调函数
    //ContrastAndBright(g_nContrastValue, 0);
    //ContrastAndBright(g_nBrightValue, 0);

    cv::waitKey(0);

    return 0;
    }
}

int MainWindow::btnzao(int value)
{



    dst =src.clone();
    int i, j, n = 500;
    n=value;
    for (int k = 0; k < n; k++)
    {
        i = rand() % src.cols;
        j = rand() % src.rows;

        if (dst.type() == CV_8UC1)
            dst.at<uchar>(j, i) = 255;
        else  if(dst.type() == CV_8UC3)
        {
            dst.at<cv::Vec3b>(j, i)[0] = 255;
            dst.at<cv::Vec3b>(j, i)[1] = 255;
            dst.at<cv::Vec3b>(j, i)[2] = 255;
        }

        else {QMessageBox message(QMessageBox::Information, "Sorry", "We Can't supported this format!", QMessageBox::Yes, this);
            message.exec();return 0;}


    }
    //cv::namedWindow("zaoimg");
    //imshow("zaoimg", src);
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    return 0;
}

int MainWindow::btnwave()
{


    cv::Mat srcX(src.rows, src.cols, CV_32F);//映射参数
    cv::Mat srcY(src.rows, src.cols, CV_32F);

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            //（i,j）像素的新位置
            srcX.at<float>(i, j) = j;//列不变
            srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);//将第i行元素按正弦移动
        }
    cv::remap(src, dst, srcX, srcY, cv::INTER_LINEAR);//进行重映射
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    //cv::namedWindow("img2");
    //imshow("img2", result);
    return 0;
}

int MainWindow::btncolorReduce(int value)
{
    int div;
    div = value;


    dst = src.clone();
    int n1 = src.rows;//行数
    int nc = src.cols*src.channels();//每行元素个数
    for (int j = 0; j < n1; j++)
    {
        uchar* data = dst.ptr<uchar>(j);//取得行j的地址
        for (int i = 0; i < nc; i++)
        {
            data[i] = data[i] / div*div + div / 2;//处理每个元素
        }//一行结束
    }
    //cv::namedWindow("图像减色");
    //imshow("图像减色", dst);
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    return 0;
}

int MainWindow::btnface()//人脸识别
{
    int status=0;
    /*if (ui->radiocamera->isChecked())
        status = 0;
    else
        status = 1;*/
    cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
    cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
    /*cv::String window_name = "Capture-Face detection";
    cv::RNG rng(12345);
    CvCapture* capture;*/
    cv::Mat frame;

    //加载级联分类器文件
    if (!face_cascade.load(face_cascade_name))
    {
        printf("Error");
        return -1;
    }
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        printf("Error");
        return -1;
    }

    if (status == 1)//读取视频
    {
        if (!path.isEmpty()) {
            cv::VideoCapture camera((const char *)path.toLocal8Bit());
            while (videostatus)
            {
                camera >> frame;
                //对当前帧进行检测
                if (!frame.empty()) { detectAndDisplay(frame); }
                int c = cv::waitKey(10);
                if (!videostatus)
                {
                    camera.release();
                    cvDestroyWindow("Edge Map");
                }
            }
        }
    }
    else//打开内置摄像头视频流
    {
        cv::VideoCapture camera(0);
        ui->label_10->hide();
        while (videostatus)
        {
            camera >> frame;
            //对当前帧进行检测
            if (!frame.empty()) { detectAndDisplay(frame); }
            int c = cv::waitKey(10);
            if (!videostatus)
            {
                camera.release();
                cvDestroyWindow("Edge Map");
            }
        }
    }
    videostatus = 1;//摄像头与视频读取状态归1
    return 0;
}





void MainWindow::on_toolButton_15_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else {
        ui->verticalSlider_8->hide();
        ui->verticalSlider_9->hide();
        ui->label_9->hide();
        ui->spinBox_4->hide();
        ui->verticalSlider_7->show();
        ui->verticalSlider_7->setMaximum(100);
        ui->label_41->show();
        ui->label_41->setText("Canny\351\230\210\345\200\274");
        ui->label_41->move(35,30);
        ui->spinBox_3->show();
        btnCanny();
    }
}

void MainWindow::on_toolButton_16_clicked()
{
    ui->label_25->hide();
    ui->label_26->move(30,374);
    ui->label_26->resize(33,26);
    ui->label_26->setText("膨胀");
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_2->setChecked(true);
    }
}

void MainWindow::on_toolButton_17_clicked()
{
    ui->stackedWidget_2->setCurrentIndex(2);
    ui->stackedWidget_3->setCurrentIndex(1);
    ui->stackedWidget_5->setCurrentIndex(0);
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_3->setChecked(true);
    }
    else btnlight();

}


void MainWindow::on_pushBtnover_clicked()
{
    videostatus = 0;

}

void MainWindow::on_toolButton_20_clicked()
{
    ui->verticalSlider_5->hide();
    ui->label_38->hide();
    ui->spinBox->hide();


    ui->pushButton->show();
    ui->pushButton->setChecked(false);
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
}

void MainWindow::on_toolButton_21_clicked()
{



      if(src.empty()) {
          QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
          message.exec();
          ui->radioButton->setChecked(true);
      }
      else{
          ui->verticalSlider_5->show();
          ui->label_38->show();
          ui->spinBox->show();

          ui->pushButton->hide();
          silderstatus=1;
          ui->verticalSlider_5->setMaximum(50);
          ui->spinBox->setMaximum(50);
      }

}

void MainWindow::on_toolButton_22_clicked()
{



    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else{
        ui->verticalSlider_5->show();
        ui->label_38->show();
        ui->spinBox->show();

        ui->pushButton->hide();
        silderstatus=2;
        ui->verticalSlider_5->setMaximum(50);
        ui->spinBox->setMaximum(50);
    }

}

void MainWindow::on_toolButton_23_clicked()
{


    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else{
        ui->verticalSlider_5->show();
        ui->label_38->show();
        ui->spinBox->show();

        ui->pushButton->hide();
        silderstatus=3;
        ui->verticalSlider_5->setMaximum(50);
        ui->spinBox->setMaximum(50);
    }

}

void MainWindow::on_toolButton_24_clicked()
{


    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else{
        ui->verticalSlider_5->show();
        ui->label_38->show();
        ui->spinBox->show();

        ui->pushButton->hide();
        silderstatus=4;
        ui->verticalSlider_5->setMaximum(50);
        ui->spinBox->setMaximum(50);
    }

}

void MainWindow::on_toolButton_25_clicked()
{

    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else{
        ui->verticalSlider_5->show();
        ui->label_38->show();
        ui->spinBox->show();

        ui->pushButton->hide();
        silderstatus=5;
        ui->verticalSlider_5->setMaximum(50);
        ui->spinBox->setMaximum(50);
    }

}

void MainWindow::on_toolButton_19_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();

    }
    else{
        ui->label_3->setText(path);
        ui->label_6->setPixmap(QPixmap::fromImage(*img));
        ui->label_6->resize(img->width(), img->height());
        ui->scrollArea_2->setWidget(ui->label_6);

        ui->label_36->setPixmap(QPixmap::fromImage(*img));
        ui->label_36->resize(img->width(), img->height());
        ui->scrollArea_7->setWidget(ui->label_36);

        ui->label_37->setPixmap(QPixmap::fromImage(*img));
        ui->label_37->resize(img->width(), img->height());
        ui->scrollArea_12->setWidget(ui->label_37);

}
}

void MainWindow::on_pushBtnmax_clicked()
{
    close();
}

void MainWindow::on_pushButton_31_clicked()
{
    showMinimized();
}



void MainWindow::on_toolButton_27_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_2->setChecked(true);
    }
    else{
        ui->stackedWidget_5->setCurrentIndex(1);
        ui->stackedWidget_3->setCurrentIndex(3);
        ui->pushButton_11->hide();
        ui->pushButton_15->hide();
    }
}

void MainWindow::on_toolButton_28_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_6->setChecked(true);
    }else{
        ui->stackedWidget_5->setCurrentIndex(1);
        ui->pushButton_11->hide();
        ui->pushButton_15->hide();
        ui->addAlpha->setEnabled(false);
        ui->addBeta->setEnabled(false);
    }
}

void MainWindow::on_toolButton_29_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_6->setChecked(true);
    }else{
        ui->stackedWidget_5->setCurrentIndex(1);
        ui->pushButton_11->hide();
        ui->pushButton_15->hide();
        ui->addAlpha->setEnabled(false);
        ui->addBeta->setEnabled(false);
    }
}

void MainWindow::on_verticalSlider_5_valueChanged(int value)
{



    switch (silderstatus)
    {


    case 1:
    {
        cv::boxFilter(src,dst, -1, cv::Size(value + 1,value + 1));
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
        break;
    case 2:
    {
        cv::blur(src,dst, cv::Size(value + 1,value + 1),cv::Point(-1,-1));
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
        break;
    case 3:
    {
        cv::GaussianBlur(src,dst,cv::Size(value*2 + 1,value*2 + 1),0,0);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
        break;
    case 4:
    {
        cv::medianBlur(src,dst,value*2+1 );
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
        break;
    case 5:
    {
        cv::bilateralFilter(src,dst,value,value*2,value/2);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
        break;

    case 6:
    {
        btnzao(value);
    }
        break;
    }
}

void MainWindow::on_toolButton_6_clicked()
{

   ui->verticalSlider_5->show();
   ui->spinBox->show();
   ui->label_38->show();
   ui->label_38->setText("\345\206\205\346\240\270\345\200\274");



   if(src.empty()) {
       QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
       message.exec();
       ui->radioButton_2->setChecked(true);
   }
   else {silderstatus=6;ui->verticalSlider_5->setMaximum(10000);ui->spinBox->setMaximum(10000);}




}

void MainWindow::on_toolButton_30_clicked()
{
    ui->stackedWidget_3->setCurrentIndex(2);
    ui->stackedWidget_5->setCurrentIndex(0);
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_3->setChecked(true);
    }
}



void MainWindow::on_verticalSlider_valueChanged(int value)
{
    if(ui->toolButton_17->isChecked())
        ContrastAndBright(value,ui->verticalSlider_2->value());
}

void MainWindow::on_verticalSlider_2_valueChanged(int value)
{
    if(ui->toolButton_17->isChecked())
        ContrastAndBright(ui->verticalSlider->value(),value);
}

void MainWindow::on_toolButton_18_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(5);
    ui->stackedWidget_4->setCurrentIndex(5);
    ui->stackedWidget_5->setCurrentIndex(0);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);
}

void MainWindow::on_toolButton_32_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }

}

void MainWindow::on_toolButton_33_clicked()
{
    ui->stackedWidget_3->setCurrentIndex(0);
    ui->stackedWidget_5->setCurrentIndex(0);
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_3->setChecked(true);
    }
}

void MainWindow::on_verticalSlider_6_valueChanged(int value)
{
    if(ui->toolButton_33->isChecked())
        btncolorReduce(value);
}

void MainWindow::on_dial_2_valueChanged(int value)
{
    if(ui->toolButton_30->isChecked())imageScaling(value);
}

void MainWindow::on_dial_valueChanged(int value)
{
    if(ui->toolButton_30->isChecked())imageTurn(value);
}

void MainWindow::on_verticalSlider_7_valueChanged(int value)
{
    if(ui->toolButton_15->isChecked()||ui->toolButton_8->isChecked()){
        if(ui->toolButton_15->isChecked()){ui->verticalSlider_7->setMaximum(100);CannyThreshold(value,0);}
        else btnSobel(value);

    }

}








void MainWindow::Relief()
{

    src.copyTo(dst);
   for (int y = 1; y < src.rows - 1; y++)
    {
        uchar *p0 = src.ptr<uchar>(y);//指向第y行首元素指针
        uchar *p1 = src.ptr<uchar>(y + 1);//指向第y+1行首元素指针

        uchar *rut = dst.ptr<uchar>(y);
        for (int x = 1; x < src.cols - 1; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                rut[3 * x + c] =cv::saturate_cast<int>( p1[3 * (x + 1) + c] - p0[3 * (x - 1) + c] + 128);
            }
        }
    }
}

void MainWindow::Expand()
{
    src.copyTo(dst);
    int R1 = sqrtf(src.cols*src.cols + src.rows*src.rows) / 2;//直接关系到放大力度，与R1成正比
    cv::Point center(src.cols / 2, src.rows / 2);
    for (int y = 0; y < src.rows; y++)
    {
        uchar *p_dst = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; x++)
        {
            int dis = norm(cv::Point(x, y) - center);
            if (dis < R1)
            {
                int newX = (x - center.x)*dis / R1 + center.x;
                int newY = (y - center.y)*dis / R1 + center.y;

                p_dst[3 * x] = src.at<uchar>(newY, newX * 3);
                p_dst[3 * x + 1] = src.at<uchar>(newY, newX * 3 + 1);
                p_dst[3 * x + 2] = src.at<uchar>(newY, newX * 3 + 2);
            }
        }
    }
}

void MainWindow::Squeezing()
{

    src.copyTo(dst);
    cv::Point center(src.cols / 2, src.rows / 2);
    for (int y = 0; y < src.rows; y++)
    {
        uchar *p_dst = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; x++)
        {
            double theta = atan2((double)(y - center.y), (double)(x - center.x));
            int R2 = sqrtf(norm(cv::Point(x, y) - center)) * 8;//关系到挤压力度，与R2成反比
            int newX = center.x + (int)(R2*cos(theta));
            int newY = center.y + (int)(R2*sin(theta));

            if (newX < 0)
                newX = 0;
            else if (newX >= src.cols)
                newX = src.cols - 1;
            if (newY < 0)
                newY = 0;
            else if (newY >= src.rows)
                newY = src.rows - 1;

            p_dst[3 * x] = src.at<uchar>(newY, newX * 3);
            p_dst[3 * x + 1] = src.at<uchar>(newY, newX * 3 + 1);
            p_dst[3 * x + 2] = src.at<uchar>(newY, newX * 3 + 2);
        }
    }

}

void MainWindow::ColorMap()
{

    src.copyTo(dst);
    cv::Mat imgColor[12];
    cv::Mat display(src.rows * 3, src.cols * 4, CV_8UC3);
    cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    for (int i = 0; i < 12; i++)
    {
        applyColorMap(dst, imgColor[i], i);
        int x = i % 4;
        int y = i / 4;
        cv::Mat displayROI = display(cv::Rect(x*src.cols, y*src.rows, src.cols, src.rows));
        cv::resize(imgColor[i], displayROI, displayROI.size());
    }
    dst = display;
}

void MainWindow::WaveSin()
{

    src.copyTo(dst);
    double angle = 0.0;
    int delta = 10;//周期
    int A = 10;//振幅
    for (int y = 0; y < src.rows; y++)
    {
        int changeX = A*sin(angle);
        uchar *srcP = src.ptr<uchar>(y);
        uchar *dstP = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; x++)
        {
            if (changeX + x < src.cols&&changeX + x>0)
            {
                dstP[3 * x] = srcP[3 * (x + changeX)];
                dstP[3 * x + 1] = srcP[3 * (x + changeX) + 1];
                dstP[3 * x + 2] = srcP[3 * (x + changeX) + 2];
            }
            else if (x <= changeX)
            {
                dstP[3 * x] = srcP[0];
                dstP[3 * x + 1] = srcP[1];
                dstP[3 * x + 2] = srcP[2];
            }
            else if (x >= src.cols - changeX)
            {
                dstP[3 * x] = srcP[3 * (src.cols - 1)];
                dstP[3 * x + 1] = srcP[3 * (src.cols - 1) + 1];
                dstP[3 * x + 2] = srcP[3 * (src.cols - 1) + 2];
            }
        }
        angle += ((double)delta) / 100;
    }
}

void Blending(cv::Mat &src, cv::Mat &src2, double alpha, double beta)//对两个图像做加权求和
{
    cv::Mat result;
    if (src.size == src2.size&&src.channels() == src.channels())//图片的大小和通道须一致
    {
        cv::addWeighted(src, alpha, src2, beta, 0, result);
    }
    else
        printf("error");
    cv::namedWindow("Blending", cv::WINDOW_NORMAL);
    cv::imshow("Blending", result);

}

void MainWindow::on_toolButton_38_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }
    else Relief();
}



void MainWindow::on_toolButton_40_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }
    else Expand();
}

void MainWindow::on_toolButton_41_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }
    else Squeezing();
}

void MainWindow::on_toolButton_42_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }
    else ColorMap();
}

void MainWindow::on_toolButton_43_clicked()
{
    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_5->setChecked(true);
    }
    else WaveSin();
}



void MainWindow::on_verticalSlider_3_valueChanged(int value)
{
    if(ui->toolButton_16->isChecked())
    {
        int offset=value-5;
        int Absolute_offset=offset>0?offset:-offset;
        cv::Mat element=cv::getStructuringElement(elementshape,cv::Size(Absolute_offset*2+1,Absolute_offset*2+1),cv::Point(Absolute_offset,Absolute_offset));
        if(offset<0)
            cv::morphologyEx(src,dst,cv::MORPH_ERODE,element);
        else
            cv::morphologyEx(src,dst,cv::MORPH_DILATE,element);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if(ui->toolButton_26->isChecked())
    {
        int offset=value-5;
        int Absolute_offset=offset>0?offset:-offset;
        cv::Mat element=cv::getStructuringElement(elementshape,cv::Size(Absolute_offset*2+1,Absolute_offset*2+1),cv::Point(Absolute_offset,Absolute_offset));
        if(offset<0)
            cv::morphologyEx(src,dst,cv::MORPH_OPEN,element);
        else
            cv::morphologyEx(src,dst,cv::MORPH_CLOSE,element);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }
    else if (ui->toolButton_31->isChecked())
    {
        int offset=value-5;
        int Absolute_offset=offset>0?offset:-offset;
        cv::Mat element=cv::getStructuringElement(elementshape,cv::Size(Absolute_offset*2+1,Absolute_offset*2+1),cv::Point(Absolute_offset,Absolute_offset));
        if(offset<0)
            cv::morphologyEx(src,dst,cv::MORPH_TOPHAT,element);
        else
            cv::morphologyEx(src,dst,cv::MORPH_BLACKHAT,element);
        ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
    }

}





void MainWindow::on_verticalSlider_4_valueChanged(int value)
{
    elementshape=value;
}

void MainWindow::on_toolButton_7_clicked()
{

    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else{
        ui->verticalSlider_8->hide();
        ui->verticalSlider_9->hide();
        ui->label_9->hide();
        ui->spinBox_4->hide();
        ui->verticalSlider_7->hide();
        ui->label_41->hide();
        ui->spinBox_3->hide();
        btnScharr();

    }
}

void MainWindow::on_toolButton_9_clicked()
{
    ui->label_10->show();
    btnface();
}

void MainWindow::on_pushButton_26_clicked()
{
    if(help==NULL)
       {
           help=new MainWindow1(NULL);
       }

       help->show();
}





void MainWindow::on_toolButton_8_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else {
        ui->verticalSlider_8->hide();
        ui->verticalSlider_9->hide();
        ui->label_9->hide();
        ui->spinBox_4->hide();
        ui->verticalSlider_7->show();
        ui->verticalSlider_7->setMaximum(3);
        ui->label_41->show();
        ui->label_41->setText("Sobel\351\230\210\345\200\274");
        ui->label_41->move(35,30);
        ui->spinBox_3->show();
        btnSobel(1);
    }
}

void MainWindow::on_toolButton_10_clicked()
{
    ui->verticalSlider_5->hide();
    ui->label_38->hide();
    ui->spinBox->hide();

    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else {ui->pushButton->show();ui->pushButton->setChecked(false);}
}

void MainWindow::on_toolButton_11_clicked()
{
    ui->verticalSlider_5->hide();
    ui->label_38->hide();
    ui->spinBox->hide();


    if(src.empty()) {
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton->setChecked(true);
    }
    else {ui->pushButton->show();ui->pushButton->setChecked(false);}
}


void MainWindow::on_toolButton_26_clicked()
{
    ui->label_26->move(18,374);
    ui->label_26->resize(45,26);
    ui->label_26->setText("开运算");
    ui->label_25->show();
    ui->label_25->move(22,30);
    ui->label_25->setText("\351\227\255\350\277\220\347\256\227");
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_2->setChecked(true);
    }
}

void MainWindow::on_toolButton_31_clicked()
{
    ui->label_26->move(30,374);
    ui->label_26->resize(33,26);
    ui->label_26->setText("顶帽");
    ui->label_25->show();
    ui->label_25->setText("黑帽");
    ui->label_25->move(30,30);
    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_2->setChecked(true);
    }
}



void MainWindow::on_toolButton_14_clicked()
{
    ui->stackedWidget_5->setCurrentIndex(1);
    ui->pushButton_11->show();
    ui->pushButton_15->show();
    ui->addAlpha->setEnabled(true);
    ui->addBeta->setEnabled(true);

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_6->setChecked(true);
    }
}

void MainWindow::on_toolButton_34_clicked()
{


    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else{
    ui->verticalSlider_8->show();
    ui->verticalSlider_9->show();
    ui->label_9->show();
    ui->spinBox_4->show();
    ui->label_41->show();
    ui->label_41->setText("模式");
    ui->label_41->move(78,30);
    ui->verticalSlider_7->hide();
    ui->spinBox_3->hide();
    }
}

void MainWindow::on_verticalSlider_9_valueChanged(int value)
{
    ThresholdType=value;
    cv::cvtColor(src,dst,cv::COLOR_BGR2GRAY);
}

void MainWindow::on_verticalSlider_8_valueChanged(int value)
{
    cv::threshold(dst,dst,value,255,ThresholdType);
    ui->label_6->setPixmap(QPixmap::fromImage(cvMat2QImage(dst)));
}


void MainWindow::on_toolButton_12_clicked()
{
    ui->toolButton_19->show();
    ui->pushBtnchange->show();
    ui->pushButton_7->show();
    ui->stackedWidget_2->setCurrentIndex(1);
    ui->stackedWidget_4->setCurrentIndex(1);
    ui->stackedWidget_5->setCurrentIndex(1);
    ui->pushBtnopen->setEnabled(true);
    ui->pushButton_14->setEnabled(true);
    ui->radioButton_2->setChecked(true);
    ui->radioButton_3->setChecked(true);
    ui->radioButton_4->setChecked(true);
    ui->radioButton_5->setChecked(true);
}

void MainWindow::on_open_clicked()
{
    openFile();
}

void MainWindow::on_pushButton_4_clicked()
{
    QString path1;
    path=":/images/exm1.jpg";
    path1="exm1.jpg";
    src = cv::imread((const char *)path1.toLocal8Bit());
    if( img->load(path) ) //加载图像
    {

        ui->frame->hide();

        ui->label_3->setText(path);
        ui->label_6->setPixmap(QPixmap::fromImage(*img));
        ui->label_6->resize(img->width(), img->height());
        ui->scrollArea_2->setWidget(ui->label_6);

        ui->label_36->setPixmap(QPixmap::fromImage(*img));
        ui->label_36->resize(img->width(), img->height());
        ui->scrollArea_7->setWidget(ui->label_36);
        ui->label_37->setPixmap(QPixmap::fromImage(*img));
        ui->label_37->resize(img->width(), img->height());
        ui->scrollArea_12->setWidget(ui->label_37);
    }else{
        QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


        //delete img;
        return;
    }
}

void MainWindow::on_pushButton_5_clicked()
{
    QString path1;
    path=":/images/exm2.jpg";
    path1="exm2.jpg";
    src = cv::imread((const char *)path1.toLocal8Bit());
    if( img->load(path) ) //加载图像
    {

        ui->frame->hide();

        ui->label_3->setText(path);
        ui->label_6->setPixmap(QPixmap::fromImage(*img));
        ui->label_6->resize(img->width(), img->height());
        ui->scrollArea_2->setWidget(ui->label_6);

        ui->label_36->setPixmap(QPixmap::fromImage(*img));
        ui->label_36->resize(img->width(), img->height());
        ui->scrollArea_7->setWidget(ui->label_36);
        ui->label_37->setPixmap(QPixmap::fromImage(*img));
        ui->label_37->resize(img->width(), img->height());
        ui->scrollArea_12->setWidget(ui->label_37);
    }else{
        QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


        //delete img;
        return;
    }
}

void MainWindow::on_pushButton_6_clicked()
{
    QString path1;
    path=":/images/exm3.jpg";
    path1="exm3.jpg";
    src = cv::imread((const char *)path1.toLocal8Bit());
    if( img->load(path) ) //加载图像
    {

        ui->frame->hide();

        ui->label_3->setText(path);
        ui->label_6->setPixmap(QPixmap::fromImage(*img));
        ui->label_6->resize(img->width(), img->height());
        ui->scrollArea_2->setWidget(ui->label_6);

        ui->label_36->setPixmap(QPixmap::fromImage(*img));
        ui->label_36->resize(img->width(), img->height());
        ui->scrollArea_7->setWidget(ui->label_36);
        ui->label_37->setPixmap(QPixmap::fromImage(*img));
        ui->label_37->resize(img->width(), img->height());
        ui->scrollArea_12->setWidget(ui->label_37);
    }else{
        QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


        //delete img;
        return;
    }
}




void MainWindow::on_addAlpha_valueChanged(double value)
{
    if(src.size()!=dst.size()){
        QMessageBox::warning(this,tr("叠加图像失败"), tr("此功能需要选取两张相同尺寸的图像！"));
    }else{
    Alpha = value;
    Blending(src,dst,Alpha,Beta);
    }
}

void MainWindow::on_addBeta_valueChanged(double value)
{
    if(src.size()!=dst.size()){
        QMessageBox::warning(this,tr("叠加图像失败"), tr("此功能需要选取两张相同尺寸的图像！"));
    }else{
    Beta = value;
    Blending(src,dst,Alpha,Beta);
    }
}

void MainWindow::on_pushButton_11_clicked()
{
    path = QFileDialog::getOpenFileName(this,tr("打开文件"), "/", tr("Images (*.ico *.png *.bmp *.jpg *.tif *.GIF *.mp4 *.MPG)"));

    if(!path.isEmpty())
    {
                src=cv::imread((const char *)path.toLocal8Bit());


                if(! ( img->load(path) ) ) //加载图像
                {
                    QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));


                    delete img;
                    return;
                }
    }

    ui->label_36->setPixmap(QPixmap::fromImage(*img));
    ui->label_36->resize(img->width(), img->height());
    ui->scrollArea_7->setWidget(ui->label_36);
    ui->pushButton_11->hide();


}

void MainWindow::on_pushButton_15_clicked()
{
    path = QFileDialog::getOpenFileName(this,tr("打开文件"), "/", tr("Images (*.ico *.png *.bmp *.jpg *.tif *.GIF *.mp4 *.MPG)"));

    if(!path.isEmpty())
    {
                dst=cv::imread((const char *)path.toLocal8Bit());


                if(! ( img2.load(path) ) ) //加载图像
                {
                    QMessageBox::warning(this,tr("打开图像失败"), tr("不能打开图像：\n%1").arg(path));



                    return;
                }
    }
    if(src.size()!=dst.size()){
        QMessageBox::warning(this,tr("叠加图像失败"), tr("此功能需要选取两张相同尺寸的图像！"));
    }else{
    ui->label_37->setPixmap(QPixmap::fromImage(img2));
    ui->label_37->resize(img2.width(), img2.height());
    ui->scrollArea_12->setWidget(ui->label_37);
    ui->pushButton_15->hide();
    }
}

void MainWindow::on_toolButton_35_clicked()
{

    if(src.empty()){
        QMessageBox message(QMessageBox::Information, "提示", "您还没有选择图片!", QMessageBox::Yes, this);
        message.exec();
        ui->radioButton_4->setChecked(true);
    }
    else{
    ui->verticalSlider_8->hide();
    ui->verticalSlider_9->hide();
    ui->label_9->hide();
    ui->spinBox_4->hide();
    ui->verticalSlider_7->hide();
    ui->spinBox_3->hide();
    ui->label_41->hide();
    Waterfloodfill *waterflood=new Waterfloodfill(src);
    waterflood->start();
    }
}


