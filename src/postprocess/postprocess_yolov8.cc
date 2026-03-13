// YOLOv8/YOLOv10/YOLO11 postprocess - single output [1, 84, 8400] format
// Based on Ultralytics export format: 4 bbox + 80 class scores per proposal

#include "postprocess_common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>

static char* labels[OBJ_CLASS_NUM];
static int labels_loaded = 0;

static char* readLine(FILE* fp, char* buffer, int* len);
static int readLines(const char* fileName, char* lines[], int max_line);
static int loadLabelName(const char* locationFilename, char* label[]);
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1);
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold);
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);

static char* readLine(FILE* fp, char* buffer, int* len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;
    buffer = (char*)malloc(buff_len + 1);
    if (!buffer) return NULL;
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void* tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL;
        }
        buffer = (char*)tmp;
        buffer[i++] = (char)ch;
    }
    buffer[i] = '\0';
    *len = (int)buff_len;
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char* fileName, char* lines[], int max_line)
{
    FILE* file = fopen(fileName, "r");
    char* s = NULL;
    int i = 0;
    int n = 0;
    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char* locationFilename, char* label[])
{
    if (labels_loaded) return 0;
    printf("loadLabelName %s\n", locationFilename);
    int ret = readLines(locationFilename, label, OBJ_CLASS_NUM);
    if (ret > 0) labels_loaded = 1;
    return ret < 0 ? -1 : 0;
}

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

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
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

/* YOLOv8 single output: [1, 84, 8400] - 84 = 4(bbox) + 80(class), 8400 = num_boxes */
extern "C"
int postprocess_yolov8(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path)
{
    if (!rknn_process || !rknn_process->outputs || rknn_process->io_num.n_output < 1)
        return -1;
    if (loadLabelName(label_path, labels) < 0)
        return -1;

    memset(group, 0, sizeof(detect_result_group_t));

    int model_in_h = rknn_process->model_height;
    int model_in_w = rknn_process->model_width;
    int32_t zp = qnt_zps.size() > 0 ? qnt_zps[0] : 0;
    float scale = qnt_scales.size() > 0 ? qnt_scales[0] : 1.0f;

    int8_t* out_ptr = (int8_t*)rknn_process->outputs[0].buf;
    int n_channel, n_boxes;
    if (rknn_process->output_attrs[0].n_dims >= 3) {
        /* NCHW: [1, 84, 8400] - channel=84, boxes=8400 */
        n_channel = rknn_process->output_attrs[0].dims[1];
        n_boxes = rknn_process->output_attrs[0].dims[2];
    } else {
        n_channel = 84;
        n_boxes = rknn_process->output_attrs[0].n_elems / n_channel;
    }

    int num_classes = n_channel - 4;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    int8_t thres_i8 = (int8_t)(box_conf_threshold / scale + zp);
    if (thres_i8 < -128) thres_i8 = -128;
    if (thres_i8 > 127) thres_i8 = 127;

    for (int i = 0; i < n_boxes; i++) {
        float cx = deqnt_affine_to_f32(out_ptr[0 * n_boxes + i], zp, scale);
        float cy = deqnt_affine_to_f32(out_ptr[1 * n_boxes + i], zp, scale);
        float w = deqnt_affine_to_f32(out_ptr[2 * n_boxes + i], zp, scale);
        float h = deqnt_affine_to_f32(out_ptr[3 * n_boxes + i], zp, scale);

        int max_cls = 0;
        int8_t max_val = out_ptr[4 * n_boxes + i];
        for (int c = 1; c < num_classes; c++) {
            int8_t v = out_ptr[(4 + c) * n_boxes + i];
            if (v > max_val) { max_val = v; max_cls = c; }
        }
        if (max_val < thres_i8) continue;

        float conf = deqnt_affine_to_f32(max_val, zp, scale);
        float x1 = (cx - w / 2);
        float y1 = (cy - h / 2);
        float bw = w;
        float bh = h;

        filterBoxes.push_back(x1);
        filterBoxes.push_back(y1);
        filterBoxes.push_back(bw);
        filterBoxes.push_back(bh);
        objProbs.push_back(conf);
        classId.push_back(max_cls);
    }

    int validCount = (int)objProbs.size();
    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++) indexArray.push_back(i);
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());
    for (int c : class_set)
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);

    int last_count = 0;
    float scale_w = rknn_process->scale_w;
    float scale_h = rknn_process->scale_h;
    BOX_RECT pads = rknn_process->pads;

    for (int i = 0; i < validCount && last_count < OBJ_NUMB_MAX_SIZE; i++) {
        if (indexArray[i] == -1) continue;
        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = objProbs[i];
        if (labels[id])
            strncpy(group->results[last_count].name, labels[id], OBJ_NAME_MAX_SIZE - 1);
        group->results[last_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        last_count++;
    }
    group->count = last_count;
    return 0;
}

extern "C" void deinit_postprocess_yolov8(void)
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i]) { free(labels[i]); labels[i] = NULL; }
    }
    labels_loaded = 0;
}
