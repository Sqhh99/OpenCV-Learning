#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "opencvtools.h"
#include <QTemporaryFile>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , scene(new QGraphicsScene(this))
    , timer(new QTimer(this))
{
    ui->setupUi(this);

    connect(ui->startAct, &QAction::triggered, this, &MainWindow::onStartAct);
    connect(ui->closeAct, &QAction::triggered, this, &MainWindow::onCloseAct);
    connect(ui->ImageChooseBtn,&QPushButton::clicked, this, &MainWindow::onImageChooseBtn);
    connect(ui->ShowImageBtn, &QPushButton::clicked, this, &MainWindow::onImageShowBtn);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrameGraphicsView);
    connect(ui->VideoChooseBtn, &QPushButton::clicked, this, &MainWindow::onVideoChooseBtn);
    connect(ui->ShowVideoBtn, &QPushButton::clicked, this, &MainWindow::onVideoShowBtn);
    connect(ui->stopBtn, &QPushButton::clicked, this, &MainWindow::onStopBtn);
    connect(ui->cvtColorBtn, &QPushButton::clicked, this, &MainWindow::onCvtColorBtn);
    connect(ui->GaussianBlurBtn, &QPushButton::clicked, this, &MainWindow::onGaussianBlurBtn);
    connect(ui->dilateBtn, &QPushButton::clicked, this, &MainWindow::onDilateBtn);
    connect(ui->erodeBtn, &QPushButton::clicked, this, &MainWindow::onErodeBtn);
    connect(ui->CannyBtn, &QPushButton::clicked, this, &MainWindow::onCannyBtn);
    connect(ui->rShowBtn, &QPushButton::clicked, this, &MainWindow::onrShowBtn);
    connect(ui->resizeBtn, &QPushButton::clicked, this, &MainWindow::onresizeBtn);
    connect(ui->CropBtn, &QPushButton::clicked, this, &MainWindow::onCropBtn);
    connect(ui->DrawShapesAndTextBtn, &QPushButton::clicked, this, &MainWindow::onDrawShapesAndTextBtn);
    connect(ui->WarpImagesBtn, &QPushButton::clicked, this, &MainWindow::onWarpImagesBtn);
    connect(ui->ColorDetectionBtn, &QPushButton::clicked, this, &MainWindow::onColorDetectionBtn);
    connect(ui->ShapeContourDetectionBtn, &QPushButton::clicked, this, &MainWindow::onShapeContourDetectionBtn);
    connect(ui->FaceDetectionBtn, &QPushButton::clicked, this, &MainWindow::onFaceDetectionBtn);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateFrameGraphicsView()
{
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
        timer->stop();
        qDebug() << "Video finished!";
        return;
    }
    // imshow("Image", frame);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);


    QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qimg);

    scene->clear();
    scene->addPixmap(pixmap);
    scene->setSceneRect(0, 0, pixmap.width(), pixmap.height());
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);
}

void MainWindow::onStartAct()
{
    std::string pb_file_path = "./opencv_face_detector_uint8.pb";
    std::string pbtxt_file_path = "./opencv_face_detector.pbtxt";
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(pb_file_path, pbtxt_file_path);
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    cv::Mat frame;

    while (true)
    {
        cap.read(frame);
        if (frame.empty())
        {
            break;
        }
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));
        net.setInput(blob);

        cv::Mat probs = net.forward();

        cv::Mat detectMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

        for (int row = 0; row < detectMat.rows; row++)
        {
            float conf = detectMat.at<float>(row, 2);
            if (conf > 0.5)
            {
                float x1 = detectMat.at<float>(row, 3) * frame.cols;
                float y1 = detectMat.at<float>(row, 4) * frame.rows;
                float x2 = detectMat.at<float>(row, 5) * frame.cols;
                float y2 = detectMat.at<float>(row, 6) * frame.rows;
                cv::Rect box(x1, y1, x2 - x1, y2 - y1);
                cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
            }
        }
        cv::imshow("Opencv4.9 DNN", frame);
        char c = cv::waitKey(1);
        if (c == 27)
        {
            break;
        }
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void MainWindow::onCloseAct()
{
    exit(0);
}

void MainWindow::onImageChooseBtn()
{
    QString filePath = QFileDialog::getOpenFileName(this,
                                                    tr("选择图像文件"),
                                                    QDir::homePath(),
                                                    tr("图像文件 (*.png *.jpg *.jpeg *.bmp *.gif)"));
    if (!filePath.isEmpty()) ui->ImagePathEdit->setText(filePath);
}

void MainWindow::onVideoChooseBtn()
{
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "Select Video File",
        "",
        "Video Files (*.mp4 *.avi *.mkv)"
        );
    if (!filePath.isEmpty()) ui->VideoPathEdit->setText(filePath);
}

void MainWindow::onImageShowBtn()
{
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "文件有问题");
        return;
    }
    // cv::imshow("Image",mat);
    QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888);
    QPixmap pixmap = QPixmap::fromImage(image);
    scene->clear();
    scene->addPixmap(pixmap);
    scene->setSceneRect(0, 0, pixmap.width(), pixmap.height());
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);
}

void MainWindow::onVideoShowBtn()
{
    QString path = ui->VideoPathEdit->toPlainText();
    if (ui->radioButton->isChecked()) {
        cap.open(0, cv::CAP_DSHOW);
    } else {
        if (path.isEmpty()) {
            QMessageBox::warning(this, "警告", "路径为空");
            return;
        }
        cap.open(path.toLocal8Bit().constData());
    }

    if (!cap.isOpened()) {
        QMessageBox::warning(this, "警告", "无法打开文件");
        return;
    }
    timer->start(33);
}

void MainWindow::onStopBtn()
{
    timer->stop();
    cap.release();
}

void MainWindow::onCvtColorBtn()
{
    cv::Mat imgGray;
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }
    // cv::imshow("Gray", imgGray);
    cv::cvtColor(mat, imgGray, cv::COLOR_BGR2GRAY);
    QImage image(imgGray.data, imgGray.cols, imgGray.rows, imgGray.step, QImage::Format_Grayscale8);
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);
}

void MainWindow::onGaussianBlurBtn()
{
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());

    // 检查图片是否加载成功
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }

    // 获取用户输入参数
    int Gx = ui->Gx->text().toInt();
    int Gy = ui->Gy->text().toInt();
    double sigmax = ui->sigmaX->text().toDouble();
    double sigmay = ui->sigmaY->text().toDouble();

    // 检查高斯核大小是否合法
    if (Gx <= 0 || Gy <= 0 || Gx % 2 == 0 || Gy % 2 == 0) {
        QMessageBox::warning(this, "警告", "高斯核必须为正奇数");
        return;
    }

    // 转换为 RGB 图像（避免显示时颜色错乱）
    cv::Mat matRGB;
    cv::cvtColor(mat, matRGB, cv::COLOR_BGR2RGB);

    // 应用高斯模糊
    cv::Mat imgGaussian;
    cv::GaussianBlur(matRGB, imgGaussian, cv::Size(Gx, Gy), sigmax, sigmay);

    // 转换为 QImage 并显示
    QImage image(imgGaussian.data, imgGaussian.cols, imgGaussian.rows, imgGaussian.step, QImage::Format_RGB888);
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);

}

void MainWindow::onDilateBtn()
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());

    // 检查图像是否加载成功
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }

    // 转换为灰度图（可选，根据需求）
    cv::Mat grayMat, imgDil;
    cv::cvtColor(mat, grayMat, cv::COLOR_BGR2GRAY);

    // 膨胀操作
    cv::dilate(grayMat, imgDil, kernel);

    // 转换为 QImage 并显示
    QImage image(imgDil.data, imgDil.cols, imgDil.rows, imgDil.step, QImage::Format_Grayscale8);
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);


}

void MainWindow::onErodeBtn()
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());

    // 检查图像是否加载成功
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }

    // 转换为灰度图（可选）
    cv::Mat grayMat, imgErode;
    cv::cvtColor(mat, grayMat, cv::COLOR_BGR2GRAY);

    // 腐蚀操作
    cv::erode(grayMat, imgErode, kernel);

    // 转换为 QImage 并显示
    QImage image(imgErode.data, imgErode.cols, imgErode.rows, imgErode.step, QImage::Format_Grayscale8);
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);

}

void MainWindow::onCannyBtn()
{
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }
    cv::Mat grayMat;
    cv::cvtColor(mat, grayMat, cv::COLOR_BGR2GRAY);
    cv::Mat imgCanny;
    double threshold1 = ui->threshold1->text().toDouble();
    double threshold2 = ui->threshold2->text().toDouble();
    cv::Canny(grayMat, imgCanny, threshold1, threshold2);
    QImage image(imgCanny.data, imgCanny.cols, imgCanny.rows, imgCanny.step, QImage::Format_Grayscale8);
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    ui->graphicsView->setScene(scene);
    ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);

}

void MainWindow::onrShowBtn()
{
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }
    cv::imshow("image", mat);
    ui->currentSizeLab->setText(QString("当前大小：%1 x %2").arg(mat.size().width).arg(mat.size().height));
}

void MainWindow::onresizeBtn()
{
    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    cv::Mat resizeM;
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }
    int reHeightSBox = ui->reHeightSBox->value();
    int reWidthSBox = ui->reWidthSBox->value();
    double ScaleX = ui->ScaleX->value();
    double ScaleY = ui->ScaleY->value();

    if (reHeightSBox == 0 || reWidthSBox == 0) {
        cv::resize(mat, resizeM, cv::Size(), ScaleX, ScaleY);
    } else {
        cv::resize(mat, resizeM, cv::Size(reHeightSBox, reWidthSBox));
    }
    cv::imshow("resize",resizeM);

}

void MainWindow::onCropBtn()
{
    int CropXSBox = ui->CropXSBox->value();
    int CropYSBox = ui->CropYSBox->value();
    int CropHeightSBox = ui->CropHeightSBox->value();
    int CropWidthSBox = ui->CropWidthSBox->value();

    QString path = ui->ImagePathEdit->toPlainText();
    cv::Mat mat = cv::imread(path.toLocal8Bit().constData());
    cv::Mat resizeM;
    if (mat.empty()) {
        QMessageBox::warning(this, "警告", "请选择图片");
        return;
    }

    cv::Rect roi(CropXSBox, CropYSBox, CropWidthSBox, CropHeightSBox);
    cv::Mat imgCrop = mat(roi);

    cv::imshow("Crop", imgCrop);
}

void MainWindow::onDrawShapesAndTextBtn()
{
    // Blank Image
    cv::Mat img(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));

    circle(img, cv::Point(256, 256), 155, cv::Scalar(0, 69, 255),cv::FILLED);
    rectangle(img, cv::Point(130, 226), cv::Point(382, 286), cv::Scalar(255, 255, 255), cv::FILLED);
    line(img, cv::Point(130, 296), cv::Point(382, 296), cv::Scalar(255, 255, 255), 2);

    putText(img, "Murtaza's Workshop", cv::Point(137, 262), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 69, 255),2);

    imshow("Image", img);
}

void MainWindow::onWarpImagesBtn()
{
    QFile file(":/Resources/Resources/cards.jpg");
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }
    QByteArray imageData = file.readAll();
    QImage image;
    image.loadFromData(imageData);
    cv::Mat img = OpencvTools::QImageToCvMat(image);
    cv::Mat matrix, imgWarp;
    float w = 250, h = 350;

    cv::Point2f src[4] = { {529,142},{771,190},{405,395},{674,457} };
    cv::Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

    matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, cv::Point(w, h));

    for (int i = 0; i < 4; i++)
    {
        circle(img, src[i], 10, cv::Scalar(0, 0, 255), cv::FILLED);
        putText(img, std::to_string(i+1), src[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    }

    imshow("Image", img);
    imshow("Image Warp", imgWarp);
}

// 全局变量，用于存储滑动条的值
int hmin = 0, smin = 110, vmin = 153;
int hmax = 19, smax = 240, vmax = 255;

// 回调函数，用于处理滑动条的值变化
void onTrackbarChange(int, void*) {
    // 该函数在滑动条值变化时被调用
}

void MainWindow::onColorDetectionBtn()
{
    using namespace cv;
    using namespace std;
    QFile file(":/Resources/Resources/lambo.png");
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }
    QByteArray imageData = file.readAll();
    QImage image;
    image.loadFromData(imageData);
    cv::Mat img = OpencvTools::QImageToCvMat(image);
    cv::Mat imgHSV, mask;

    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    cv::namedWindow("Trackbars", WINDOW_AUTOSIZE);
    cv::createTrackbar("Hue Min", "Trackbars", nullptr, 179, onTrackbarChange);
    cv::createTrackbar("Hue Max", "Trackbars", nullptr, 179, onTrackbarChange);
    cv::createTrackbar("Sat Min", "Trackbars", nullptr, 255, onTrackbarChange);
    cv::createTrackbar("Sat Max", "Trackbars", nullptr, 255, onTrackbarChange);
    cv::createTrackbar("Val Min", "Trackbars", nullptr, 255, onTrackbarChange);
    cv::createTrackbar("Val Max", "Trackbars", nullptr, 255, onTrackbarChange);

    // 设置滑动条的初始值
    setTrackbarPos("Hue Min", "Trackbars", hmin);
    setTrackbarPos("Hue Max", "Trackbars", hmax);
    setTrackbarPos("Sat Min", "Trackbars", smin);
    setTrackbarPos("Sat Max", "Trackbars", smax);
    setTrackbarPos("Val Min", "Trackbars", vmin);
    setTrackbarPos("Val Max", "Trackbars", vmax);

    while (true) {
        // 获取滑动条的当前值
        hmin = getTrackbarPos("Hue Min", "Trackbars");
        hmax = getTrackbarPos("Hue Max", "Trackbars");
        smin = getTrackbarPos("Sat Min", "Trackbars");
        smax = getTrackbarPos("Sat Max", "Trackbars");
        vmin = getTrackbarPos("Val Min", "Trackbars");
        vmax = getTrackbarPos("Val Max", "Trackbars");

        cv::Scalar lower(hmin, smin, vmin);
        cv::Scalar upper(hmax, smax, vmax);
        cv::inRange(imgHSV, lower, upper, mask);

        cv::imshow("Image", img);
        cv::imshow("Image HSV", imgHSV);
        cv::imshow("Image Mask", mask);

        // 处理 Qt 事件
        QCoreApplication::processEvents();

        // 等待 1 毫秒
        if (cv::waitKey(1) >= 0) {
            break; // 如果按下任意键，退出循环
        }
    }

    cv::destroyAllWindows();
}

void MainWindow::onShapeContourDetectionBtn()
{
    QFile file(":/Resources/Resources/shapes.png");
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }
    QByteArray imageData = file.readAll();
    QImage image;
    image.loadFromData(imageData);
    cv::Mat img = OpencvTools::QImageToCvMat(image);
    cv::Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

    // Preprocessing
    cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, cv::Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    dilate(imgCanny, imgDil, kernel);

    OpencvTools::getContours(imgDil,img);

    imshow("Image", img);
}

void MainWindow::onFaceDetectionBtn()
{
    QFile file(":/Resources/Resources/test.png");
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }
    QByteArray imageData = file.readAll();
    QImage image;
    image.loadFromData(imageData);
    cv::Mat img = OpencvTools::QImageToCvMat(image);

    QFile xmlFile(":/Resources/Resources/haarcascade_frontalface_default.xml");
    if (!xmlFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "警告", "无法打开 XML 文件");
        return;
    }
    QTemporaryFile tempFile;
    if (!tempFile.open()) {
        QMessageBox::warning(this, "警告", "无法创建临时文件");
        return;
    }
    tempFile.write(xmlFile.readAll());
    tempFile.close();

    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(tempFile.fileName().toStdString())) {
        QMessageBox::warning(this, "警告", "无法加载 XML 文件");
        return;
    }
    if (faceCascade.empty()) { QMessageBox::warning(this, "警告", "XML file not loaded"); return;}

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(img, faces, 1.1, 10);

    for (int i = 0; i < faces.size(); i++)
    {
        rectangle(img, faces[i].tl(), faces[i].br(), cv::Scalar(255, 0, 255), 3);
    }

    imshow("Image", img);
}


