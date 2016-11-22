#-------------------------------------------------
#
# Project created by QtCreator 2016-03-05T11:00:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets\
                                        multimedia \
                                    multimediawidgets

TARGET = TextEditor
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    mainwindow1.cpp \
    clickablelabel.cpp \
    waterfloodfill.cpp


HEADERS  += \
    mainwindow.h \
    mainwindow1.h \
    clickablelabel.h \
    waterfloodfill.h


FORMS    += mainwindow.ui \
    mainwindow1.ui
INCLUDEPATH +=   E:\opencv\opencv3.0\build\include\
  E:\opencv\opencv3.0\build\include\opencv\
 E:\opencv\opencv3.0\build\include\opencv2
CONFIG(debug,debug|release){
LIBS +=  E:\opencv\opencv3.0\build\x64\vc12\lib\opencv_ts300d.lib\
 E:\opencv\opencv3.0\build\x64\vc12\lib\opencv_world300d.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_core300d.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_highgui300d.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_imgproc300d.lib\
}else{
LIBS +=  E:\opencv\opencv3.0\build\x64\vc12\lib\opencv_ts300.lib\
 E:\opencv\opencv3.0\build\x64\vc12\lib\opencv_world300.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_core300.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_highgui300.lib\
 E:\opencv\opencv3.0\build\x64\vc12\staticlib\opencv_imgproc300.lib\
}
RESOURCES += \
    menu.qrc

DISTFILES +=
RC_FILE +=icon.rc
