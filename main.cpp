#include "mainwindow.h"
#include <QSplashScreen>
#include <QPixmap>
#include <QDebug>
#include <QElapsedTimer>
#include <QDateTime>


#include <QApplication>
#include"QTextCodec"

int main(int argc, char *argv[])
{
    //解决库路径问题============info@seatrix.com
        QTextCodec *xcodec = QTextCodec::codecForLocale() ;
        QString exeDir = xcodec->toUnicode( QByteArray(argv[0]) ) ;
        QString BKE_CURRENT_DIR = QFileInfo(exeDir).path() ;
        QStringList  libpath;

        libpath << BKE_CURRENT_DIR+QString::fromLocal8Bit("/plugins/platforms");
        libpath << BKE_CURRENT_DIR <<BKE_CURRENT_DIR+QString::fromLocal8Bit("/plugins/imageformats");
        libpath << BKE_CURRENT_DIR+QString::fromLocal8Bit("/plugins");
        libpath << QApplication::libraryPaths();
        QApplication::setLibraryPaths(libpath) ;
    //=========================

    QApplication a(argc, argv);
        QPixmap pixmap(":/images/启动界面(1).png");
        QSplashScreen screen(pixmap);
        screen.show();
        screen.showMessage("", Qt::AlignBottom, Qt::black);
    #if 0
        int delayTime = 10;
        QElapsedTimer timer;
        timer.start();
        while(timer.elapsed() < (delayTime * 1000))
        {
             a.processEvents();
        }
    #endif

    #if 0
        QDateTime n=QDateTime::currentDateTime();
        QDateTime now;
        do{
            now=QDateTime::currentDateTime();
            a.processEvents();
        } while (n.secsTo(now)<=10);//5为需要延时的秒数
    #endif
    MainWindow w;
    w.show();
    screen.finish(&w);
    return a.exec();
}
