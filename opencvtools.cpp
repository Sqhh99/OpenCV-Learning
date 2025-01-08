#include "opencvtools.h"

cv::Mat OpencvTools::QImageToCvMat(const QImage &image) {
    cv::Mat mat;
    switch (image.format()) {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    case QImage::Format_RGBA8888: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    case QImage::Format_RGB888: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    }
    case QImage::Format_Indexed8: {
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    default:
        qWarning() << "QImage format not supported!";
        break;
    }
    return mat;
}

void OpencvTools::getContours(cv::Mat imgDil, cv::Mat img) {

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL,
                 cv::CHAIN_APPROX_SIMPLE);
    // drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    std::vector<std::vector<cv::Point>> conPoly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);
        std::string objectType;

        if (area > 1000) {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            boundRect[i] = boundingRect(conPoly[i]);

            int objCor = (int)conPoly[i].size();

            if (objCor == 3) {
                objectType = "Tri";
            } else if (objCor == 4) {
                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                if (aspRatio > 0.95 && aspRatio < 1.05) {
                    objectType = "Square";
                } else {
                    objectType = "Rect";
                }
            } else if (objCor > 4) {
                objectType = "Circle";
            }

            drawContours(img, conPoly, i, cv::Scalar(255, 0, 255), 2);
            rectangle(img, boundRect[i].tl(), boundRect[i].br(),
                      cv::Scalar(0, 255, 0), 5);
            putText(img, objectType, {boundRect[i].x, boundRect[i].y - 5},
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 69, 255), 2);
        }
    }
}
