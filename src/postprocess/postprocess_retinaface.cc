// RetinaFace postprocess - face detection + 5 landmarks
// Outputs: boxes [1,4200,4], scores [1,4200,2], landmarks [1,4200,10]
// Class 0=background, 1=face

#include "postprocess_common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1);
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold);
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0f);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0f);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
        (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) continue;
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) continue;
            float xmin0 = outputLocations[n * 4 + 0], ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = xmin0 + outputLocations[n * 4 + 2], ymax0 = ymin0 + outputLocations[n * 4 + 3];
            float xmin1 = outputLocations[m * 4 + 0], ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = xmin1 + outputLocations[m * 4 + 2], ymax1 = ymin1 + outputLocations[m * 4 + 3];
            if (CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1) > threshold)
                order[j] = -1;
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
    if (left >= right) return left;
    float key = input[left];
    int key_index = indices[left];
    int low = left, high = right;
    while (low < high) {
        while (low < high && input[high] <= key) high--;
        input[low] = input[high]; indices[low] = indices[high];
        while (low < high && input[low] >= key) low++;
        input[high] = input[low]; indices[high] = indices[low];
    }
    input[low] = key; indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
    return low;
}

extern "C"
int postprocess_retinaface(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path)
{
    (void)label_path;
    if (!rknn_process || rknn_process->io_num.n_output < 2) return -1;
    memset(group, 0, sizeof(detect_result_group_t));

    int num_anchors = rknn_process->output_attrs[0].n_elems / 4;
    if (num_anchors > 10000) num_anchors = 4200;
    int32_t box_zp = qnt_zps.size() > 0 ? qnt_zps[0] : 0;
    float box_scale = qnt_scales.size() > 0 ? qnt_scales[0] : 1.0f;
    int32_t score_zp = qnt_zps.size() > 1 ? qnt_zps[1] : 0;
    float score_scale = qnt_scales.size() > 1 ? qnt_scales[1] : 1.0f;

    int8_t* box_ptr = (int8_t*)rknn_process->outputs[0].buf;
    int8_t* score_ptr = (int8_t*)rknn_process->outputs[1].buf;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    for (int idx = 0; idx < num_anchors; idx++) {
        float conf = deqnt_affine_to_f32(score_ptr[idx * 2 + 1], score_zp, score_scale);
        if (conf < box_conf_threshold) continue;

        float x = deqnt_affine_to_f32(box_ptr[idx], box_zp, box_scale);
        float y = deqnt_affine_to_f32(box_ptr[idx + num_anchors], box_zp, box_scale);
        float w = deqnt_affine_to_f32(box_ptr[idx + num_anchors * 2], box_zp, box_scale);
        float h = deqnt_affine_to_f32(box_ptr[idx + num_anchors * 3], box_zp, box_scale);

        float x1 = x - w / 2;
        float y1 = y - h / 2;
        filterBoxes.push_back(x1);
        filterBoxes.push_back(y1);
        filterBoxes.push_back(w);
        filterBoxes.push_back(h);
        objProbs.push_back(conf);
        classId.push_back(0);
    }

    int validCount = (int)objProbs.size();
    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++) indexArray.push_back(i);
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    nms(validCount, filterBoxes, classId, indexArray, 0, nms_threshold);

    int last = 0;
    float sw = rknn_process->scale_w, sh = rknn_process->scale_h;
    BOX_RECT pads = rknn_process->pads;
    int mh = rknn_process->model_height, mw = rknn_process->model_width;

    for (int i = 0; i < validCount && last < OBJ_NUMB_MAX_SIZE; i++) {
        if (indexArray[i] == -1) continue;
        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float bw = filterBoxes[n * 4 + 2], bh = filterBoxes[n * 4 + 3];
        group->results[last].box.left = (int)(clamp(x1, 0, mw) / sw);
        group->results[last].box.top = (int)(clamp(y1, 0, mh) / sh);
        group->results[last].box.right = (int)(clamp(x1 + bw, 0, mw) / sw);
        group->results[last].box.bottom = (int)(clamp(y1 + bh, 0, mh) / sh);
        group->results[last].prop = objProbs[i];
        strncpy(group->results[last].name, "face", OBJ_NAME_MAX_SIZE - 1);
        group->results[last].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        last++;
    }
    group->count = last;
    return 0;
}

extern "C" void deinit_postprocess_retinaface(void) { (void)0; }
