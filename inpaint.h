
#ifndef INPAINT_H
#define INPAINT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct InpaintData {
    Mat inputImage, mask, updatedMask, result, workImage;
    Mat sourceRegion, targetRegion, originalSourceRegion;
    Mat gradientX, gradientY, confidence, data;
    Mat LAPLACIAN_KERNEL, NORMAL_KERNELX, NORMAL_KERNELY;
    Point2i bestMatchUpperLeft, bestMatchLowerRight;
    vector<Point> fillFront;
    vector<Point2f> normals;

    int halfPatchWidth = 4;
    int mode = 1;
    int targetIndex = 0;
};


void initializeMats(InpaintData& data);
void calculateGradients(InpaintData& data);


void computeFillFront(InpaintData& data);
void computeConfidence(InpaintData& data);
void computeData(InpaintData& data);
void computeTarget(InpaintData& data);
void computeBestPatch(InpaintData& data);
void updateMats(InpaintData& data);
bool checkEnd(const InpaintData& data);


void getPatch(const InpaintData& data, const Point2i& centerPixel, Point2i& upperLeft, Point2i& lowerRight);
void inpaint(InpaintData& data);

#endif
