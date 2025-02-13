#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

// ฟังก์ชันคำนวณค่า Hu Moments
double getHuMoment(vector<Point>& contour)
{
    Moments mu = moments(contour);
    double hu[7];
    HuMoments(mu, hu);
    return hu[0];
}

// โครงสร้างข้อมูลติดตามผู้คน
vector<vector<Point>> trails; // เก็บเส้น Trail ของแต่ละคน
vector<int> objectIDs;        // เก็บ ID ของวัตถุที่ตรวจจับได้ในเฟรมก่อนหน้า
const int MAX_TRAIL_LENGTH = 30; // จำกัดความยาวของเส้น Trail
const int MAX_DISTANCE = 50;  // ระยะที่อนุญาตให้วัตถุเคลื่อนที่ระหว่างเฟรม (ใช้จับคู่ ID)

int main(int argc, char** argv)
{
    Mat fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    VideoCapture cap(argv[1]); // เปิดกล้องหรือไฟล์วิดีโอ
    if (!cap.isOpened()) return -1;

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
        threshold(fgMaskMOG2, fgMaskMOG2, 220, 255, 0);

        // ทำ Morphological Operations
        Mat erodedMask, dilatedMask;
        erode(fgMaskMOG2, erodedMask, Mat(), Point(-1, -1), 2);
        dilate(erodedMask, dilatedMask, Mat(), Point(-1, -1), 3);

        // หาขอบเขต (Contours)
        vector<vector<Point>> contours;
        findContours(dilatedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Point> newCenters;
        for (size_t i = 0; i < contours.size(); i++)
        {
            if (contourArea(contours[i]) < 250) continue; // กรอง Noise

            // คำนวณ Bounding Box
            Rect bbox = boundingRect(contours[i]);
            int padding = 10;
            bbox.x = max(bbox.x - padding, 0);
            bbox.y = max(bbox.y - padding, 0);
            bbox.width = min(bbox.width + (2 * padding), frame.cols - bbox.x);
            bbox.height = min(bbox.height + (2 * padding), frame.rows - bbox.y);

            // คำนวณค่า Hu Moment
            double huMoment = getHuMoment(contours[i]);

            // คำนวณจุดศูนย์กลาง Bounding Box
            Point objectCenter = Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            newCenters.push_back(objectCenter);

            // วาด Bounding Box
            rectangle(frame, bbox, Scalar(0, 255, 0), 2);

            // แสดงค่า Hu Moment บนภาพ
            stringstream ss;
            ss << "Hu : " << huMoment;
            putText(frame, ss.str(), Point(bbox.x, bbox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        }

        // เชื่อมโยง Object ID กับศูนย์กลางของวัตถุใหม่
        vector<int> newObjectIDs(newCenters.size(), -1);
        vector<vector<Point>> newTrails(newCenters.size());

        for (size_t i = 0; i < newCenters.size(); i++)
        {
            int bestMatch = -1;
            double bestDistance = MAX_DISTANCE;

            for (size_t j = 0; j < objectIDs.size(); j++)
            {
                double dist = norm(newCenters[i] - trails[j].back());
                if (dist < bestDistance)
                {
                    bestMatch = j;
                    bestDistance = dist;
                }
            }

            if (bestMatch != -1)
            {
                newObjectIDs[i] = objectIDs[bestMatch];
                newTrails[i] = trails[bestMatch];
            }
            else
            {
                newObjectIDs[i] = objectIDs.empty() ? 0 : (*max_element(objectIDs.begin(), objectIDs.end()) + 1);
            }

            newTrails[i].push_back(newCenters[i]);
            if (newTrails[i].size() > MAX_TRAIL_LENGTH)
                newTrails[i].erase(newTrails[i].begin());
        }

        // อัปเดตค่า Trails และ Object IDs
        trails = newTrails;
        objectIDs = newObjectIDs;

        // วาดเส้น Trail สำหรับทุกคน
        for (const auto& trail : trails)
        {
            for (size_t i = 1; i < trail.size(); i++)
            {
                line(frame, trail[i - 1], trail[i], Scalar(0, 0, 255), 2);
            }
        }

        imshow("FG Mask MOG 2", dilatedMask);
        imshow("frame with Bounding Box", frame);

        if (waitKey(20) >= 0) break;
    }

    return 0;
}
