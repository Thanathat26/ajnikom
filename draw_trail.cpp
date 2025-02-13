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

// ฟังก์ชันสร้างสีแบบสุ่มแต่คงที่สำหรับแต่ละ ID
Scalar getColorById(int id) {
    RNG rng(id * 999); // ใช้ ID เป็น seed เพื่อให้ได้สีเดิมเสมอสำหรับ ID เดียวกัน
    return Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

// โครงสร้างข้อมูลติดตามผู้คน
vector<vector<Point>> trails; // เก็บเส้น Trail ของแต่ละคน
vector<int> objectIDs;        // เก็บ ID ของวัตถุที่ตรวจจับได้ในเฟรมก่อนหน้า
const int MAX_TRAIL_LENGTH = 30; // จำกัดความยาวของเส้น Trail
const int MAX_DISTANCE = 50;  // ระยะที่อนุญาตให้วัตถุเคลื่อนที่ระหว่างเฟรม (ใช้จับคู่ ID)

int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    Mat fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    VideoCapture cap(argv[1]); // เปิดกล้องหรือไฟล์วิดีโอ
    if (!cap.isOpened()) {
        cout << "Error: Could not open video file" << endl;
        return -1;
    }

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
        vector<Rect> newBboxes;
        vector<double> huMoments;
        
        for (size_t i = 0; i < contours.size(); i++)
        {
            if (contourArea(contours[i]) < 250) continue;

            Rect bbox = boundingRect(contours[i]);
            int padding = 10;
            bbox.x = max(bbox.x - padding, 0);
            bbox.y = max(bbox.y - padding, 0);
            bbox.width = min(bbox.width + (2 * padding), frame.cols - bbox.x);
            bbox.height = min(bbox.height + (2 * padding), frame.rows - bbox.y);
            
            newBboxes.push_back(bbox);
            double huMoment = getHuMoment(contours[i]);
            huMoments.push_back(huMoment);
            Point objectCenter = Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            newCenters.push_back(objectCenter);
        }

        // เชื่อมโยง Object ID กับศูนย์กลางของวัตถุใหม่
        vector<int> newObjectIDs(newCenters.size(), -1);
        vector<vector<Point>> newTrails(newCenters.size());
        vector<bool> usedOldIDs(objectIDs.size(), false);  // เพิ่มการติดตาม ID ที่ถูกใช้แล้ว

        // จับคู่วัตถุกับ ID เดิมก่อน
        for (size_t i = 0; i < newCenters.size(); i++)
        {
            int bestMatch = -1;
            double bestDistance = MAX_DISTANCE;

            for (size_t j = 0; j < objectIDs.size(); j++)
            {
                if (!usedOldIDs[j])  // ตรวจสอบว่า ID นี้ยังไม่ถูกใช้
                {
                    double dist = norm(newCenters[i] - trails[j].back());
                    if (dist < bestDistance)
                    {
                        bestMatch = j;
                        bestDistance = dist;
                    }
                }
            }

            if (bestMatch != -1)
            {
                newObjectIDs[i] = objectIDs[bestMatch];
                newTrails[i] = trails[bestMatch];
                usedOldIDs[bestMatch] = true;  // มาร์คว่า ID นี้ถูกใช้แล้ว
            }
        }

        // สร้าง ID ใหม่สำหรับวัตถุที่ยังไม่มี ID
        int nextNewID = objectIDs.empty() ? 0 : (*max_element(objectIDs.begin(), objectIDs.end()) + 1);
        for (size_t i = 0; i < newCenters.size(); i++)
        {
            if (newObjectIDs[i] == -1)  // ถ้ายังไม่มี ID
            {
                newObjectIDs[i] = nextNewID++;
                newTrails[i].push_back(newCenters[i]);
            }
        }

        // อัปเดต trails สำหรับวัตถุที่มี ID แล้ว
        for (size_t i = 0; i < newCenters.size(); i++)
        {
            if (newTrails[i].empty() || 
                newTrails[i].back() != newCenters[i])  // ป้องกันการเพิ่มจุดซ้ำ
            {
                newTrails[i].push_back(newCenters[i]);
            }
            
            if (newTrails[i].size() > MAX_TRAIL_LENGTH)
            {
                newTrails[i].erase(newTrails[i].begin());
            }
        }

        // วาด Bounding Box และข้อมูล
        for (size_t i = 0; i < newCenters.size(); i++)
        {
            Scalar color = getColorById(newObjectIDs[i]);
            rectangle(frame, newBboxes[i], color, 2);
            
            // stringstream ss;
            // ss << "ID: " << newObjectIDs[i];
            // putText(frame, ss.str(), Point(newBboxes[i].x, newBboxes[i].y - 25), 
            //         FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            
            // stringstream ss_hu;
            // ss_hu << "Hu: " << huMoments[i];
            // putText(frame, ss_hu.str(), Point(newBboxes[i].x, newBboxes[i].y - 5), 
            //         FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        // อัปเดตค่า Trails และ Object IDs
        trails = newTrails;
        objectIDs = newObjectIDs;

        // วาดเส้น Trail
        for (size_t i = 0; i < trails.size(); i++)
        {
            Scalar color = getColorById(objectIDs[i]);
            for (size_t j = 1; j < trails[i].size(); j++)
            {
                line(frame, trails[i][j - 1], trails[i][j], color, 2);
            }
        }

        imshow("FG Mask MOG 2", dilatedMask);
        imshow("frame with Bounding Box", frame);

        char key = waitKey(20);
        if (key == 27) break;  // กด ESC เพื่อออก
    }

    return 0;
}