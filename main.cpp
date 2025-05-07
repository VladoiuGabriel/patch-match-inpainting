

#include <opencv2/opencv.hpp>
#include <iostream>
#include "inpaint.h"

using namespace std;
using namespace cv;

Mat image, originalImage, inpaintMask;
Point prevPt(-1, -1);
int thickness = 5;

static void onMouse(int event, int x, int y, int flags, void*) {
    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
        Point pt(x, y);
        if (prevPt.x < 0)
            prevPt = pt;
        line(inpaintMask, prevPt, pt, Scalar::all(255), thickness, 8, 0);
        line(image, prevPt, pt, Scalar::all(255), thickness, 8, 0);
        prevPt = pt;
        imshow("image", image);
    }
}

int main() {
    string filename = R"(C:\Users\Gabi\Desktop\ImageInpainting\Lincoln.jpg)";
    originalImage = imread(filename, IMREAD_COLOR);
    resize(originalImage, originalImage, Size(640, 480));

    image = originalImage.clone();
    inpaintMask = Mat::zeros(image.size(), CV_8U);
    namedWindow("image", 1);
    imshow("image", image);
    setMouseCallback("image", onMouse, 0);

    while (true) {
        char c = waitKey();

        if (c == 'e')
            break;

        if (c == 'o') {
            inpaintMask = Scalar::all(0);
            image = originalImage.clone();
            imshow("image", image);
        }

        if (c == 'i' || c == ' ') {
            InpaintData data;
            data.inputImage = originalImage.clone();
            data.mask = inpaintMask.clone();
            data.workImage = originalImage.clone();
            data.updatedMask = data.mask.clone();
            data.result.create(originalImage.size(), originalImage.type());

            inpaint(data);
            imwrite("result.bmp", data.result);
            inpaintMask = Scalar::all(0);
            namedWindow("result");
            imshow("result", data.result);
            waitKey(0);
        }

        if (c == '+') thickness++;
        if (c == '-') thickness--;

        if (thickness < 3) thickness = 3;
        if (thickness > 12) thickness = 12;
    }

    return 0;
}
