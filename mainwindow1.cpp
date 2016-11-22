#include "mainwindow1.h"
#include "ui_mainwindow1.h"
#include<QtGui>
#include<QTextStream>
#include<QDebug>
MainWindow1::MainWindow1(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow1)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::FramelessWindowHint);//去掉标题栏
    QFile file(":/help1.html");
    if(!file.open(QFile::ReadOnly | QFile::Text))
        qDebug() << "Can not open";
    QTextStream in(&file);
    ui->textBrowser->setHtml(in.readAll());
    QFile file4(":/help4.html");
    if(!file4.open(QFile::ReadOnly | QFile::Text))
        qDebug() << "Can not open";
    QTextStream in4(&file4);
    ui->textBrowser_4->setHtml(in4.readAll());

}
void MainWindow1::mousePressEvent(QMouseEvent *event)
{
    mousePosition = event->pos();
//只对标题栏范围内的鼠标事件进行处理
    if (mousePosition.x()<=0)
        return;
    if ( mousePosition.x()>=891)
        return;
    if (mousePosition.y()<=0 )
        return;
    if (mousePosition.y()>=600)
        return;
    isMousePressed = true;
}
void MainWindow1::mouseMoveEvent(QMouseEvent *event)
{
    if ( isMousePressed==true )
    {
        QPoint movePot = event->globalPos() - mousePosition;
        move(movePot);
    }
}
void MainWindow1::mouseReleaseEvent(QMouseEvent *)
{
    isMousePressed=false;
}

void MainWindow1::paintEvent (QPaintEvent* )
{
    QPainter painter(this);
    QPen pen(Qt::black);
    pen.setWidth(2);
    painter.setPen(pen);
    painter.drawRect(0,0,891,600);
}


MainWindow1::~MainWindow1()
{
    delete ui;
}

void MainWindow1::on_toolBtnhelp_1_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow1::on_toolBtnhelp_2_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow1::on_toolBtnhelp_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow1::on_toolBtnhelp_4_clicked()
{
    ui->stackedWidget->setCurrentIndex(3);
}

void MainWindow1::on_toolBtnhelp_5_clicked()
{
    ui->stackedWidget->setCurrentIndex(4);
}

void MainWindow1::on_toolBtnhelp_6_clicked()
{
    ui->stackedWidget->setCurrentIndex(5);
}

void MainWindow1::on_pushBtnmin_h_clicked()
{
    showMinimized();
}

void MainWindow1::on_pushBtnmax_h_clicked()
{
    close();
}
