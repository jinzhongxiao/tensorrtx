#pragma once
#include "config.h"

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
    float keypoints[51];  // 17*3 keypoints
};

struct AffineMatrix {
    float value[6];
};
struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};
const int bbox_element =
        sizeof(AffineMatrix) / sizeof(float) + 1;  // left, top, right, bottom, confidence, class, keepflag
