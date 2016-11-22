#ifndef MAINWINDOW1_H
#define MAINWINDOW1_H

#include <QMainWindow>
#include<QFile>

namespace Ui {
class MainWindow1;
}

class MainWindow1 : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow1(QWidget *parent = 0);
    ~MainWindow1();
protected:
    void paintEvent (QPaintEvent* );

    void mousePressEvent(QMouseEvent *event);

    void mouseMoveEvent(QMouseEvent *event);

    void mouseReleaseEvent(QMouseEvent *);

private slots:
    void on_toolBtnhelp_1_clicked();

    void on_toolBtnhelp_2_clicked();

    void on_toolBtnhelp_3_clicked();

    void on_toolBtnhelp_4_clicked();

    void on_toolBtnhelp_5_clicked();

    void on_toolBtnhelp_6_clicked();

    void on_pushBtnmin_h_clicked();

    void on_pushBtnmax_h_clicked();



private:
    Ui::MainWindow1 *ui;
    QPoint mousePosition;
    bool isMousePressed;
};


#endif // MAINWINDOW1_H
