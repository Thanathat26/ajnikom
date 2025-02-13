#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

// ฟังก์ชันคำนวณค่า Hu Moments
double getHuMoment(vector<Point>& contour)
{
    Moments mu = moments(contour); // คำนวณโมเมนต์ของวัตถุ
    double hu[7];
    HuMoments(mu, hu); // คำนวณค่า Hu Moments
    
    return hu[0]; // คืนค่า Hu Moment ตัวแรก
}

int main(int argc, char** argv)
{
    Mat fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    VideoCapture cap(argv[1]); // เปิดกล้องหรือไฟล์วิดีโอ
    if (!cap.isOpened())
        return -1;

    namedWindow("frame with Bounding Box");
    namedWindow("FG Mask MOG 2");

    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // ใช้ Background Subtractor
        pMOG2->apply(frame, fgMaskMOG2);

        // Threshold กรอง Noise
        threshold(fgMaskMOG2, fgMaskMOG2, 250, 255, 0);

        // ทำ Morphological Operations
        Mat erodedMask, dilatedMask;
        erode(fgMaskMOG2, erodedMask, Mat(), Point(-1, -1), 5);
        dilate(erodedMask, dilatedMask, Mat(), Point(-1, -1), 7);

        // หาขอบเขต (Contours)
        vector<vector<Point>> contours;
        findContours(dilatedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++)
        {
            if (contourArea(contours[i]) < 8500) continue; // กรอง Noise

            // คำนวณ Bounding Box
            Rect bbox = boundingRect(contours[i]);
            int padding = 10;
            bbox.x = max(bbox.x - padding, 0);
            bbox.y = max(bbox.y - padding, 0);
            bbox.width = min(bbox.width + (2 * padding), frame.cols - bbox.x);
            bbox.height = min(bbox.height + (2 * padding), frame.rows - bbox.y);

            // คำนวณค่า Hu Moment
            double huMoment = getHuMoment(contours[i]);

            // วาด Bounding Box
            rectangle(frame, bbox, Scalar(0, 255, 0), 2);

            // แสดงค่า Hu Moment บนภาพ
            stringstream ss;
            ss << "Hu : " << huMoment;
            putText(frame, ss.str(), Point(bbox.x, bbox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        }

        imshow("FG Mask MOG 2", dilatedMask);
        imshow("frame with Bounding Box", frame);

        if (waitKey(20) >= 0) break;
    }

    return 0;
}
