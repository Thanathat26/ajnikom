#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat image = imread(argv[1]);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    // ใช้ Sobel Gradient
    Mat grad_x, grad_y;
    Sobel(gray_image, grad_x, CV_16S, 1, 0, 3);
    Sobel(gray_image, grad_y, CV_16S, 0, 1, 3);

    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);

    Mat gradient;
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient);

    // Threshold เพื่อตรวจจับขอบ
    Mat thresholded;
    threshold(gradient, thresholded, 50, 255, THRESH_BINARY);

    // หา Contours
    vector<vector<Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point2f center;
    Rect boundingBox;
    for (size_t i = 0; i < contours.size(); i++)
    {
        Rect tempBox = boundingRect(contours[i]);
        
        if (tempBox.area() > 10000)  
        {
            boundingBox = tempBox;
            center = Point2f(boundingBox.x + boundingBox.width / 2, 
                             boundingBox.y + boundingBox.height / 2);

            rectangle(image, boundingBox, Scalar(0, 255, 0), 2);
            circle(image, center, 5, Scalar(0, 0, 255), -1);
        }
    }

        
    if (boundingBox.area() > 10000)
    {
        int maxRadius = min(boundingBox.width, boundingBox.height) / 2;
        Mat polarImage;
        warpPolar(gray_image, polarImage, Size(360, maxRadius), center, maxRadius, INTER_LINEAR + WARP_FILL_OUTLIERS);
        //  หา roi 
        int sidecan = 30; 
        Mat leftcut = Mat::zeros(polarImage.rows, sidecan, CV_8UC1);
        Mat rightcut = polarImage(Rect(polarImage.cols - sidecan, 0, sidecan, polarImage.rows));
        Mat Roi;
        hconcat(leftcut,rightcut,Roi);
        // namedWindow("Side Edges", 1);
        // imshow("Side Edges", Roi);
        //  หมุนภาพ
        float angle = 90;
        
        Point2f rotationCenter(Roi.cols / 2.0, Roi.rows / 2.0);
        Mat rotationMatrix = getRotationMatrix2D(rotationCenter, angle, 1.0);
        Mat rotate_roi;
        cout << "ก: " << Roi.cols << endl;
        cout << "ย: " << Roi.rows << endl;
        //หนุนเเล้วใช้ roisize ภาพเลยโดนตังต้องสร้างrect ที่มีขนาดเท่ากับ ภาพ หลังจากปลิ้นขนาดมาเเล้ว ได้กว้าง 100 ยาว 354 ก็สร้าง rect ที่ขนาดตรงข้ามมาใส่
        Size rotatesize(100, 360);
        Rect bbox = RotatedRect(rotationCenter, rotatesize, angle).boundingRect();
        rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - rotationCenter.x;
        rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - rotationCenter.y;

        warpAffine(Roi,rotate_roi,rotationMatrix,bbox.size());
        namedWindow("Rotated", 1);
        imshow("Rotated", rotate_roi);

        // 1. CLAHE Equalization
Mat equalized;
Ptr<CLAHE> clahe = createCLAHE();
clahe->setClipLimit(4);
clahe->apply(rotate_roi, equalized);

// 2. คำนวณ Horizontal Projection (Histogram)
Mat projection;
reduce(equalized, projection, 0, REDUCE_SUM, CV_32SC1); // รวมความสว่างในแนวตั้ง

// 3. ปรับสเกลค่าให้ตรงกับความสูงภาพ
double minVal, maxVal;
minMaxLoc(projection, &minVal, &maxVal);

Mat histogramVis = Mat::zeros(equalized.size(), CV_8UC1); // ภาพขาวดำขนาดเดียวกับ ROI

int h = histogramVis.rows;
for (int x = 0; x < projection.cols; x++) {
    int value = projection.at<int>(x);
    int height = cvRound((value - minVal) * h / (maxVal - minVal)); // คำนวณความสูงบาร์
    line(histogramVis, Point(x, h-1), Point(x, h-1 - height), Scalar(255), 1); // วาดบาร์ขาว
}

// 4. ตรวจจับส่วนยุบตัว (Dips) ด้วย Threshold
Mat dips;
threshold(histogramVis, dips, 250, 255, THRESH_BINARY_INV); // ส่วนยุบจะมีค่าต่ำ

// 5. ประมวลผล Morphological เพื่อเน้นรอยแตก
Mat kernel = getStructuringElement(MORPH_RECT, Size(3,5));
morphologyEx(dips, dips, MORPH_CLOSE, kernel);

// แสดงผลลัพธ์
        imshow("Histogram Graph", histogramVis);

        //his 
        imshow("Polar Transform", polarImage);
    }


    
    // namedWindow("Contour Image", 1);
    // namedWindow("Gray Scale Image", 1);
    // namedWindow("Thresholded Image", 1);

    // imshow("Contour Image", image);
    // imshow("Gray Scale Image", gray_image);
    // imshow("Thresholded Image", thresholded);

    waitKey(0);
    return 0;
}