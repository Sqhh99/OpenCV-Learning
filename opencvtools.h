#ifndef OPENCVTOOLS_H
#define OPENCVTOOLS_H

#include <QApplication>
#include <QFile>
#include <QByteArray>
#include <opencv2/opencv.hpp>
#include <QDebug>

class OpencvTools
{
public:
    OpencvTools() = delete;
    OpencvTools(const OpencvTools&) = delete;
    OpencvTools& operator=(const OpencvTools&) = delete;

    cv::Mat static QImageToCvMat(const QImage &image);

    void static getContours(cv::Mat imgDil, cv::Mat img);
};

#endif // OPENCVTOOLS_H
