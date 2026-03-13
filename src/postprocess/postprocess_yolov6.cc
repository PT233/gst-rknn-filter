// YOLOv6 postprocess - 9 outputs: 3 branches x (box, score, score_sum)
// Output format: [1,4,H,W], [1,80,H,W], [1,1,H,W] per branch

#include "postprocess_common.h"
#include "postprocess_impl.h"
#include "../rknnprocess.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>

static char* labels[OBJ_CLASS_NUM];
static int labels_loaded = 0;

static int loadLabelName(const char* locationFilename, char* label[]);
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1);
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold);
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);

static char* readLine(FILE* fp, char* buffer, int* len);
static int readLines(const char* fileName, char* lines[], int max_line);

static char* readLine(FILE* fp, char* buffer, int* len) {
    int ch, i = 0;
    size_t buff_len = 0;
    buffer = (char*)malloc(1);
    if (!buffer) return NULL;
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void* tmp = realloc(buffer, buff_len + 1);
        if (!tmp) { free(buffer); return NULL; }
        buffer = (char*)tmp;
        buffer[i++] = (char)ch;
    }
    buffer[i] = '\0';
    *len = (int)buff_len;
    if (ch == EOF && (i == 0 || ferror(fp))) { free(buffer); return NULL; }
    return buffer;
}

static int readLines(const char* fileName, char* lines[], int max_line) {
    FILE* file = fopen(fileName, "r");
    char* s = NULL;
    int i = 0, n = 0;
    if (!file) return -1;
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char* locationFilename, char* label[]) {
    if (labels_loaded) return 0;
    int ret = readLines(locationFilename, label, OBJ_CLASS_NUM);
    if (ret > 0) labels_loaded = 1;
    return ret < 0 ? -1 : 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0f);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0f);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
        (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) continue;
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) continue;
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) order[j] = -1;
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices) {
    if (left >= right) return left;
    float key = input[left];
    int key_index = indices[left];
    int low = left, high = right;
    while (low < high) {
        while (low < high && input[high] <= key) high--;
        input[low] = input[high];
        indices[low] = indices[high];
        while (low < high && input[low] >= key) low++;
        input[high] = input[low];
        indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
    return low;
}

static void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[16];
        float exp_sum = 0, acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * (float)i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8_yolov6(int8_t* box_tensor, int32_t box_zp, float box_scale,
    int8_t* score_tensor, int32_t score_zp, float score_scale,
    int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId,
    float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = (int8_t)((threshold / score_scale) + score_zp);
    if (score_thres_i8 > 127) score_thres_i8 = 127;
    if (score_thres_i8 < -128) score_thres_i8 = -128;
    int8_t score_sum_thres_i8 = (int8_t)((threshold / score_sum_scale) + score_sum_zp);
    if (score_sum_thres_i8 > 127) score_sum_thres_i8 = 127;
    if (score_sum_thres_i8 < -128) score_sum_thres_i8 = -128;

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;
            int8_t max_score = -128;

            if (score_sum_tensor && score_sum_tensor[offset] < score_sum_thres_i8)
                continue;

            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                int off = offset + c * grid_len;
                if (score_tensor[off] > score_thres_i8 && score_tensor[off] > max_score) {
                    max_score = score_tensor[off];
                    max_class_id = c;
                }
            }

            if (max_score > score_thres_i8) {
                offset = i * grid_w + j;
                float box[4];
                if (dfl_len > 1) {
                    float before_dfl[16];
                    int idx = 0;
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + idx * grid_len], box_zp, box_scale);
                        idx++;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                } else {
                    for (int k = 0; k < 4; k++) {
                        box[k] = deqnt_affine_to_f32(box_tensor[offset + k * grid_len], box_zp, box_scale);
                    }
                }

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);
                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int postprocess_yolov6(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path) {
    if (rknn_process->io_num.n_output < 9) return -1;
    if (loadLabelName(label_path, labels) < 0) return -1;

    memset(group, 0, sizeof(detect_result_group_t));
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int model_in_h = rknn_process->model_height;
    int model_in_w = rknn_process->model_width;

    int output_per_branch = rknn_process->io_num.n_output / 3;
    int dfl_len = rknn_process->output_attrs[0].dims[1] / 4;
    if (dfl_len < 1) dfl_len = 1;

    for (int i = 0; i < 3; i++) {
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        int score_sum_idx = i * output_per_branch + 2;

        int grid_h = rknn_process->output_attrs[box_idx].dims[2];
        int grid_w = rknn_process->output_attrs[box_idx].dims[3];
        int stride = model_in_h / grid_h;

        void* score_sum_buf = output_per_branch >= 3 ? rknn_process->outputs[score_sum_idx].buf : nullptr;
        int32_t score_sum_zp = output_per_branch >= 3 ? qnt_zps[score_sum_idx] : 0;
        float score_sum_scale = output_per_branch >= 3 ? qnt_scales[score_sum_idx] : 1.0f;

        validCount += process_i8_yolov6(
            (int8_t*)rknn_process->outputs[box_idx].buf, qnt_zps[box_idx], qnt_scales[box_idx],
            (int8_t*)rknn_process->outputs[score_idx].buf, qnt_zps[score_idx], qnt_scales[score_idx],
            (int8_t*)score_sum_buf, score_sum_zp, score_sum_scale,
            grid_h, grid_w, stride, dfl_len,
            filterBoxes, objProbs, classId, box_conf_threshold);
    }

    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++) indexArray.push_back(i);
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    for (int i = 0; i < validCount; i++) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] - rknn_process->pads.left;
        float y1 = filterBoxes[n * 4 + 1] - rknn_process->pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = (int)(clamp(x1, 0.f, (float)model_in_w) / rknn_process->scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0.f, (float)model_in_h) / rknn_process->scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0.f, (float)model_in_w) / rknn_process->scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0.f, (float)model_in_h) / rknn_process->scale_h);
        group->results[last_count].prop = obj_conf;
        if (labels[id]) strncpy(group->results[last_count].name, labels[id], OBJ_NAME_MAX_SIZE - 1);
        group->results[last_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        last_count++;
    }
    group->count = last_count;
    return 0;
}

void deinit_postprocess_yolov6(void) {
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i]) {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
    labels_loaded = 0;
}
