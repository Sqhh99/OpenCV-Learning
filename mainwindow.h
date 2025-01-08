#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <QGraphicsScene>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private:
    cv::VideoCapture cap;

private:
    Ui::MainWindow *ui;
    QGraphicsScene* scene;
    QTimer* timer;


private slots:
    void updateFrameGraphicsView();
    void onStartAct();
    void onCloseAct();
    void onImageChooseBtn();
    void onVideoChooseBtn();
    void onImageShowBtn();
    void onVideoShowBtn();
    void onStopBtn();

private slots:
    void onCvtColorBtn();
    void onGaussianBlurBtn();
    void onDilateBtn();
    void onErodeBtn();
    void onCannyBtn();

private:
    void onrShowBtn();
    void onresizeBtn();
    void onCropBtn();
private:
    void onDrawShapesAndTextBtn();
private:
    void onWarpImagesBtn();
private:
    void onColorDetectionBtn();
private:
    void onShapeContourDetectionBtn();
private:
    void onFaceDetectionBtn();


};
#endif // MAINWINDOW_H
