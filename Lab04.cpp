#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

Mat src4;
Mat src4_gray;
int thresh4 = 100;
int max_thresh4 = 255;
RNG rng4(12345);

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>(filename_to_load);
    if (model.empty())
        cout << "Could not read the classifier " << filename_to_load << endl;

    return model;
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static float
ann_classifier(const string& filename_to_load,
    Mat data)
{
    Ptr<ANN_MLP> model;

    if (!filename_to_load.empty())
    {
        model = load_classifier<ANN_MLP>(filename_to_load);
        if (model.empty())
            return false;
    }

    float r = model->predict(data.row(0));
    printf("%f\n", r);

    return r;
}

Mat gen_feature_input(Mat& image)
{
    float max = 0;
    int val[120];

    for (int i = 0; i < image.cols; i++)
    {
        int column_sum = 0;
        for (int k = 0; k < image.rows; k++)
        {
            column_sum += image.at<unsigned char>(k, i);
        }
        val[i] = column_sum;
        if (val[i] > max) max = (float)val[i];
    }

    Mat data(1, image.cols, CV_32F);

    for (int i = 0; i < image.cols; i++)
    {
        data.at<float>(0, i) = (float)val[i] / max;
    }
    return data;
}

bool contour_features(vector<Point> ct)
{
    double area = contourArea(ct);
    RotatedRect rt = fitEllipse(ct);
    Moments mu = moments(ct);

    double hu[7];
    HuMoments(mu, hu);

    for (int i = 0; i < 7; i++)
        printf("%lf ", hu[i]);
    printf("\n");

    // ตัวอย่างเงื่อนไขใช้ hu[0] เพื่อกรอง
    if (hu[0] < 0.18) return true;
    return false;
}

int main(int argc, char** argv)
{
    Mat dst_image1;
    Mat src, src_gray, dst;
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int c;

    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image = imread(argv[1]); // Read the file
    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // เบลอภาพเพื่อลดสัญญาณรบกวน
    GaussianBlur(image, dst_image1, Size(5, 5), 0, 0);
    cvtColor(dst_image1, src_gray, COLOR_BGR2GRAY);

    // --- Sobel Edge Detection ---
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    threshold(grad, dst, 20, 255, THRESH_BINARY);

    // --- Find Contours ---
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // --- Approx & Filter Contours ---
    vector<vector<Point>> contours_poly(contours.size());
    vector<vector<Point>> contours_cycle;  // เอาไว้เก็บเฉพาะคอนทัวร์ที่ผ่านการตรวจสอบ
    vector<Rect> boundRect(contours.size());
    vector<Point2f> center(contours.size());
    vector<float> radius(contours.size());

    int j = 0;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        approxPolyDP(Mat(contours[i]), contours_poly[j], 1, true);

        // สมมติกรองเฉพาะคอนทัวร์ที่มีจำนวนจุดมากกว่า 5 (วงรี/วงกลม)
        if (contours_poly[j].size() > 5)
        {
            if (contour_features(contours_poly[j]))
            {
                contours_cycle.push_back(contours_poly[j]);
                boundRect[j] = boundingRect(Mat(contours_poly[j]));
                minEnclosingCircle(contours_poly[j], center[j], radius[j]);
            }
            j++;
        }
    }

    // --- เลือกคอนทัวร์ที่มี boundingRect ใหญ่ที่สุด ---
    int maxArea = 0;
    int max_idx = 0;

    for (int i = 0; i < j; i++)
    {
        int area = boundRect[i].width * boundRect[i].height;
        if (area > maxArea)
        {
            maxArea = area;
            max_idx = i;
        }
    }

    // --- วาดผลลัพธ์ ---
    Scalar color = Scalar(0, 255, 255); // สีเหลือง (BGR: (Blue=0, Green=255, Red=255))
    drawContours(image, contours_cycle, max_idx, color, 1, 8, vector<Vec4i>(), 0, Point());
    rectangle(image, boundRect[max_idx].tl(), boundRect[max_idx].br(), color, 2, 8, 0); 
    Mat imgOutput;
    imgOutput = image;
    cv::resize(imgOutput, imgOutput, cv::Size(), 0.5, 0.5);
    imshow("Output1", imgOutput);

    // --- Crop ส่วนที่เป็นวงกลม/วงรีที่สนใจ ---
    Mat imgCycle;
    Mat croppedDetected(image, boundRect[max_idx]);
    croppedDetected.copyTo(imgCycle);
    imshow("Cycle", imgCycle);

    // --- สร้าง Polar Transform ---
    Point2f ct;
    float rt = imgCycle.cols / 2.0f;
    ct.x = rt;
    ct.y = rt;

    Mat imgPolar;
    // WARP_FILL_OUTLIERS จะเติมขอบนอกด้วยค่าสีที่เหมาะสม
    linearPolar(imgCycle, imgPolar, ct, rt, WARP_FILL_OUTLIERS);
    /*imshow("imgPolar", imgPolar);*/

    // --- ตัดเอา edge ด้านขวาสุด 50 px มาหมุน ---
    Mat imgEdge;
    int edgeWidth = 50;
    int edgeHeight = imgPolar.rows;
    if (imgPolar.cols >= edgeWidth && imgPolar.rows >= edgeHeight)
    {
        Mat croppedEdgeRef(imgPolar, Rect(imgPolar.cols - edgeWidth, 0, edgeWidth, edgeHeight));
        croppedEdgeRef.copyTo(imgEdge);
        /*imshow("img Edge", imgEdge);*/
    }
    else {
        std::cerr << "Invalid dimensions for cropping the edge." << std::endl;
    }

    // หมุนภาพ 90 องศาทวนเข็ม
    rotate(imgEdge, imgEdge, ROTATE_90_COUNTERCLOCKWISE);
    /*imshow("imgEdge", imgEdge)*/;

    // เปลี่ยนเป็นภาพเกรย์สเกล
    cvtColor(imgEdge, imgEdge, COLOR_BGR2GRAY);
    /*imshow("imgEdge (Gray)", imgEdge);*/

    // ======================== ส่วนสร้าง Histogram =========================
    int histWidth = imgEdge.cols;
    int histHeight = 100;
    Mat imgHistogram = Mat::zeros(histHeight, histWidth, CV_8UC1);

    // สร้างเวกเตอร์สำหรับเก็บผลรวมของค่าพิกเซลในแต่ละคอลัมน์
    vector<float> hval(imgEdge.cols, 0.0f);
    float hmax = 0.0f;

    // (1) คำนวณผลรวมของแต่ละคอลัมน์ใน imgEdge
    for (int i = 0; i < imgEdge.cols; i++) {
        float sumCol = 0.0f;
        for (int j = 0; j < imgEdge.rows; j++) {
            sumCol += static_cast<float>(imgEdge.at<uchar>(j, i));
        }
        hval[i] = sumCol;
        if (sumCol > hmax) {
            hmax = sumCol;
        }
    }

    // (2) สร้าง Histogram โดย Normalize และวาดเส้นแนวตั้ง
    vector<float> colHeights(imgEdge.cols, 0.0f); // เก็บ "ความสูง normalized" ของแต่ละคอลัมน์
    for (int i = 0; i < imgEdge.cols; i++) {
        float normalizedVal = (hval[i] / hmax) * (histHeight - 1);
        colHeights[i] = normalizedVal;  // เก็บไว้เพื่อใช้เช็คเงื่อนไข

        // วาดเส้นใน imgHistogram (สีขาวบนพื้นดำ)
        line(imgHistogram,
            Point(i, histHeight - 1),
            Point(i, histHeight - 1 - cvRound(normalizedVal)),
            Scalar(255), 1);
    }

    // แสดง Histogram ปกติ
    imshow("Histogram (Full)", imgHistogram);

    // ========================= เลือกคอลัมน์ที่ความสูง < 50% =========================
    float thresholdVal = 0.6f * (histHeight - 1); // 50% ของ histHeight (99)
    vector<int> colsToKeep;
    colsToKeep.reserve(imgEdge.cols);

    // ประกาศ Mat Crack ด้านนอก
    Mat Crack;

    for (int i = 0; i < imgEdge.cols; i++) {
        if (colHeights[i] < thresholdVal) {
            colsToKeep.push_back(i);
        }
    }

    // (3) สร้างภาพใหม่ เฉพาะคอลัมน์ที่เลือก
    if (!colsToKeep.empty()) {
        // สร้าง Mat ขนาด "imgEdge.rows" x "จำนวนคอลัมน์ที่คัดเลือก"
        Mat filteredEdge(imgEdge.rows, (int)colsToKeep.size(), CV_8UC1);

        // คัดลอก column จาก imgEdge
        for (int idx = 0; idx < (int)colsToKeep.size(); idx++) {
            int colIndex = colsToKeep[idx];
            // copyTo() column จาก colIndex ของ imgEdge ไปยัง col(idx) ของ filteredEdge
            imgEdge.col(colIndex).copyTo(filteredEdge.col(idx));
        }

        // ปรับขนาดภาพ filteredEdge เป็น 120x40
        Size targetSize(120, 40);
        resize(filteredEdge, Crack, targetSize);
        equalizeHist(Crack, Crack);
        imshow("Edge (Histogram < 60%)", Crack);

    }
    else {
        std::cerr << "No columns found with histogram < 60% threshold.\n";
    }

    // prediction
    Mat feature;
    feature = gen_feature_input(Crack);

    int pre = ann_classifier("model.xml", feature);
    printf("prediction: %d\n", pre);

    Mat imgApp = Mat::zeros(Size(1024, 724), image.type());
    Mat imgOrg;
    cv::resize(image, imgOrg, cv::Size(), 0.5, 0.5);
    Rect t_rect = Rect(0, 0, imgOrg.cols, imgOrg.rows);
    imgOrg.copyTo(imgApp(t_rect));

    Mat Crack2;
    cv::resize(Crack, Crack, cv::Size(), 4, 4);
    cvtColor(Crack, Crack, COLOR_GRAY2BGR);

    Rect t_rect2 = Rect(imgOrg.cols, 0, Crack.cols, Crack.rows);
    Crack.copyTo(imgApp(t_rect2));

    if (pre == 1)
        cv::putText(imgApp, //Target img
            "NO CRACK",
            cv::Point((imgOrg.cols/2)-100, imgApp.rows-100),
            cv::FONT_HERSHEY_COMPLEX_SMALL,
            2.0,
            CV_RGB(255, 0, 0),
            2);
    else if (pre == 0)
        cv::putText(imgApp, //Target img
            "CRACK",
            cv::Point((imgOrg.cols/2)-100, imgApp.rows-100),
            cv::FONT_HERSHEY_COMPLEX_SMALL,
            2.0,
            CV_RGB(110, 120, 0),
            2);

    cv::imshow("Output", imgApp);

    waitKey(0);
    return 0;
}