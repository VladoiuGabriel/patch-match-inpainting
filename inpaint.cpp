#include "inpaint.h"
#include <cmath>

void initializeMats(InpaintData& data) {
    threshold(data.mask, data.confidence, 10, 255, THRESH_BINARY);
    threshold(data.confidence, data.confidence, 2, 1, THRESH_BINARY_INV);
    data.confidence.convertTo(data.confidence, CV_32F);

    data.sourceRegion = data.confidence.clone();
    data.sourceRegion.convertTo(data.sourceRegion, CV_8U);
    data.originalSourceRegion = data.sourceRegion.clone();

    threshold(data.mask, data.targetRegion, 10, 255, THRESH_BINARY);
    threshold(data.targetRegion, data.targetRegion, 2, 1, THRESH_BINARY);
    data.targetRegion.convertTo(data.targetRegion, CV_8U);

    data.data = Mat(data.inputImage.rows, data.inputImage.cols, CV_32F, Scalar::all(0));

    data.LAPLACIAN_KERNEL = Mat::ones(3, 3, CV_32F);
    data.LAPLACIAN_KERNEL.at<float>(1, 1) = -8;

    data.NORMAL_KERNELX = Mat::zeros(3, 3, CV_32F);
    data.NORMAL_KERNELX.at<float>(1, 0) = -1;
    data.NORMAL_KERNELX.at<float>(1, 2) = 1;
    transpose(data.NORMAL_KERNELX, data.NORMAL_KERNELY);
}

void calculateGradients(InpaintData& data) {
    Mat srcGray;
    cvtColor(data.workImage, srcGray, COLOR_BGR2GRAY);

    Scharr(srcGray, data.gradientX, CV_16S, 1, 0);
    convertScaleAbs(data.gradientX, data.gradientX);
    data.gradientX.convertTo(data.gradientX, CV_32F);

    Scharr(srcGray, data.gradientY, CV_16S, 0, 1);
    convertScaleAbs(data.gradientY, data.gradientY);
    data.gradientY.convertTo(data.gradientY, CV_32F);

    for (int x = 0; x < data.sourceRegion.cols; x++) {
        for (int y = 0; y < data.sourceRegion.rows; y++) {
            if (data.sourceRegion.at<uchar>(y, x) == 0) {
                data.gradientX.at<float>(y, x) = 0;
                data.gradientY.at<float>(y, x) = 0;
            }
        }
    }
    data.gradientX /= 255;
    data.gradientY /= 255;
}

void computeFillFront(InpaintData& data) {
    Mat sourceGradientX, sourceGradientY, boundryMat;
    filter2D(data.targetRegion, boundryMat, CV_32F, data.LAPLACIAN_KERNEL);
    filter2D(data.sourceRegion, sourceGradientX, CV_32F, data.NORMAL_KERNELX);
    filter2D(data.sourceRegion, sourceGradientY, CV_32F, data.NORMAL_KERNELY);

    data.fillFront.clear();
    data.normals.clear();

    for (int x = 0; x < boundryMat.cols; x++) {
        for (int y = 0; y < boundryMat.rows; y++) {
            if (boundryMat.at<float>(y, x) > 0) {
                data.fillFront.push_back(Point2i(x, y));

                float dx = sourceGradientX.at<float>(y, x);
                float dy = sourceGradientY.at<float>(y, x);
                Point2f normal(dy, -dx);
                float norm = std::sqrt(normal.x * normal.x + normal.y * normal.y);
                if (norm != 0) {
                    normal.x /= norm;
                    normal.y /= norm;
                }
                data.normals.push_back(normal);
            }
        }
    }
}

void computeConfidence(InpaintData& data) {
    Point2i a, b;
    for (size_t i = 0; i < data.fillFront.size(); i++) {
        getPatch(data, data.fillFront[i], a, b);
        float total = 0;
        for (int x = a.x; x <= b.x; x++) {
            for (int y = a.y; y <= b.y; y++) {
                if (data.targetRegion.at<uchar>(y, x) == 0) {
                    total += data.confidence.at<float>(y, x);
                }
            }
        }
        data.confidence.at<float>(data.fillFront[i].y, data.fillFront[i].x) = total / ((b.x - a.x + 1) * (b.y - a.y + 1));
    }
}

void computeData(InpaintData& data) {
    for (size_t i = 0; i < data.fillFront.size(); i++) {
        Point2i p = data.fillFront[i];
        Point2f n = data.normals[i];
        data.data.at<float>(p.y, p.x) = std::fabs(data.gradientX.at<float>(p.y, p.x) * n.x + data.gradientY.at<float>(p.y, p.x) * n.y) + 0.001f;
    }
}

void computeTarget(InpaintData& data) {
    float maxPriority = 0.0f;
    data.targetIndex = 0;

    for (size_t i = 0; i < data.fillFront.size(); i++) {
        Point2i p = data.fillFront[i];
        float priority = data.data.at<float>(p.y, p.x) * data.confidence.at<float>(p.y, p.x);
        if (priority > maxPriority) {
            maxPriority = priority;
            data.targetIndex = static_cast<int>(i);
        }
    }
}

bool checkEnd(const InpaintData& data) {
    for (int y = 0; y < data.sourceRegion.rows; y++) {
        for (int x = 0; x < data.sourceRegion.cols; x++) {
            if (data.sourceRegion.at<uchar>(y, x) == 0) {
                return true;
            }
        }
    }
    return false;
}

void getPatch(const InpaintData& data, const Point2i& centerPixel, Point2i& upperLeft, Point2i& lowerRight) {
    int x = centerPixel.x;
    int y = centerPixel.y;
    int minX = std::max(x - data.halfPatchWidth, 0);
    int maxX = std::min(x + data.halfPatchWidth, data.workImage.cols - 1);
    int minY = std::max(y - data.halfPatchWidth, 0);
    int maxY = std::min(y + data.halfPatchWidth, data.workImage.rows - 1);

    upperLeft = Point2i(minX, minY);
    lowerRight = Point2i(maxX, maxY);
}
void computeBestPatch(InpaintData& data) {
    double minError = 10E15, bestPatchVarience = 10E15;
    Point2i a, b;
    Point2i currentPoint = data.fillFront[data.targetIndex];
    Vec3b sourcePixel, targetPixel;
    double meanR, meanG, meanB;
    double difference, patchError;
    bool skipPatch;
    getPatch(data, currentPoint, a, b);

    int width = b.x - a.x + 1;
    int height = b.y - a.y + 1;

    int searchRadius = 40;
    int minX = std::max(0, currentPoint.x - searchRadius);
    int maxX = std::min(data.workImage.cols - width, currentPoint.x + searchRadius);
    int minY = std::max(0, currentPoint.y - searchRadius);
    int maxY = std::min(data.workImage.rows - height, currentPoint.y + searchRadius);

    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            patchError = 0;
            meanR = 0; meanG = 0; meanB = 0;
            skipPatch = false;
            int validPixelCount = 0;

            for (int x2 = 0; x2 < width; x2++) {
                for (int y2 = 0; y2 < height; y2++) {
                    if (data.originalSourceRegion.at<uchar>(y + y2, x + x2) == 0) {
                        skipPatch = true;
                        break;
                    }

                    if (x2 > 1 && x2 < width - 2 && y2 > 1 && y2 < height - 2)
                        continue;

                    if (data.sourceRegion.at<uchar>(a.y + y2, a.x + x2) == 0)
                        continue;

                    sourcePixel = data.workImage.at<Vec3b>(y + y2, x + x2);
                    targetPixel = data.workImage.at<Vec3b>(a.y + y2, a.x + x2);

                    for (int i = 0; i < 3; i++) {
                        difference = sourcePixel[i] - targetPixel[i];
                        patchError += difference * difference;
                    }

                    meanB += sourcePixel[0];
                    meanG += sourcePixel[1];
                    meanR += sourcePixel[2];
                    validPixelCount++;
                }
                if (skipPatch) break;
            }

            if (skipPatch || validPixelCount == 0) continue;

            patchError /= validPixelCount;
            meanB /= validPixelCount;
            meanG /= validPixelCount;
            meanR /= validPixelCount;

            if (patchError < minError) {
                minError = patchError;
                data.bestMatchUpperLeft = Point2i(x, y);
                data.bestMatchLowerRight = Point2i(x + width - 1, y + height - 1);

                double patchVarience = 0;
                for (int x2 = 0; x2 < width; x2++) {
                    for (int y2 = 0; y2 < height; y2++) {
                        if (data.sourceRegion.at<uchar>(a.y + y2, a.x + x2) == 0) {
                            sourcePixel = data.workImage.at<Vec3b>(y + y2, x + x2);
                            difference = sourcePixel[0] - meanB;
                            patchVarience += difference * difference;
                            difference = sourcePixel[1] - meanG;
                            patchVarience += difference * difference;
                            difference = sourcePixel[2] - meanR;
                            patchVarience += difference * difference;
                        }
                    }
                }
                bestPatchVarience = patchVarience;
            }
            else if (patchError == minError) {
                double patchVarience = 0;
                for (int x2 = 0; x2 < width; x2++) {
                    for (int y2 = 0; y2 < height; y2++) {
                        if (data.sourceRegion.at<uchar>(a.y + y2, a.x + x2) == 0) {
                            sourcePixel = data.workImage.at<Vec3b>(y + y2, x + x2);
                            difference = sourcePixel[0] - meanB;
                            patchVarience += difference * difference;
                            difference = sourcePixel[1] - meanG;
                            patchVarience += difference * difference;
                            difference = sourcePixel[2] - meanR;
                            patchVarience += difference * difference;
                        }
                    }
                }
                if (patchVarience < bestPatchVarience) {
                    minError = patchError;
                    data.bestMatchUpperLeft = Point2i(x, y);
                    data.bestMatchLowerRight = Point2i(x + width - 1, y + height - 1);
                    bestPatchVarience = patchVarience;
                }
            }
        }
    }
}

void updateMats(InpaintData& data) {
    Point2i targetPoint = data.fillFront[data.targetIndex];
    Point2i a, b;
    getPatch(data, targetPoint, a, b);
    int width = b.x - a.x + 1;
    int height = b.y - a.y + 1;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int px = a.x + x;
            int py = a.y + y;

            if (data.targetRegion.at<uchar>(py, px) == 1) {
                int sx = data.bestMatchUpperLeft.x + x;
                int sy = data.bestMatchUpperLeft.y + y;

                data.workImage.at<Vec3b>(py, px) = data.workImage.at<Vec3b>(sy, sx);
                data.gradientX.at<float>(py, px) = data.gradientX.at<float>(sy, sx);
                data.gradientY.at<float>(py, px) = data.gradientY.at<float>(sy, sx);
                data.confidence.at<float>(py, px) = data.confidence.at<float>(targetPoint.y, targetPoint.x);
                data.sourceRegion.at<uchar>(py, px) = 255;
                data.targetRegion.at<uchar>(py, px) = 0;
                data.updatedMask.at<uchar>(py, px) = 0;
            }
        }
    }
}
void inpaint(InpaintData& data) {
    initializeMats(data);
    calculateGradients(data);

    bool stay = true;
    while (stay) {
        computeFillFront(data);
        computeConfidence(data);
        computeData(data);
        computeTarget(data);
        computeBestPatch(data);
        updateMats(data);
        stay = checkEnd(data);
        waitKey(2);
    }
    data.result = data.workImage.clone();
}
