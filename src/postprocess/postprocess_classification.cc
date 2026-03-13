// Image classification postprocess - mobilenet, resnet
// Single output: [1, num_classes] - argmax for Top-1

#include "postprocess_common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static char* labels[1000];
static int labels_capacity = 1000;
static int labels_count = 0;

static char* readLine(FILE* fp, char* buffer, int* len);
static int readLines(const char* fileName, char* lines[], int max_line);

static char* readLine(FILE* fp, char* buffer, int* len)
{
    int ch, i = 0;
    size_t buff_len = 0;
    buffer = (char*)malloc(buff_len + 1);
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
    int i = 0, n = 0;
    if (!file) return -1;
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

extern "C"
int postprocess_classification(struct _RknnProcess* rknn_process,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path)
{
    if (!rknn_process || !rknn_process->outputs || rknn_process->io_num.n_output < 1)
        return -1;

    int num_classes = rknn_process->output_attrs[0].n_elems;
    if (num_classes > labels_capacity) num_classes = labels_capacity;

    if (label_path && labels_count == 0) {
        labels_count = readLines(label_path, labels, labels_capacity);
        if (labels_count < 0) labels_count = 0;
    }

    memset(group, 0, sizeof(detect_result_group_t));

    int32_t zp = qnt_zps.size() > 0 ? qnt_zps[0] : 0;
    float scale = qnt_scales.size() > 0 ? qnt_scales[0] : 1.0f;
    int8_t* out = (int8_t*)rknn_process->outputs[0].buf;

    int max_idx = 0;
    float max_val = deqnt_affine_to_f32(out[0], zp, scale);
    for (int i = 1; i < num_classes; i++) {
        float v = deqnt_affine_to_f32(out[i], zp, scale);
        if (v > max_val) { max_val = v; max_idx = i; }
    }

    group->count = 1;
    group->results[0].box.left = 10;
    group->results[0].box.top = 10;
    group->results[0].box.right = 200;
    group->results[0].box.bottom = 50;
    group->results[0].prop = max_val;
    if (max_idx < labels_count && labels[max_idx])
        strncpy(group->results[0].name, labels[max_idx], OBJ_NAME_MAX_SIZE - 1);
    else
        snprintf(group->results[0].name, OBJ_NAME_MAX_SIZE, "class_%d", max_idx);
    group->results[0].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
    return 0;
}

extern "C" void deinit_postprocess_classification(void)
{
    for (int i = 0; i < labels_count; i++) {
        if (labels[i]) { free(labels[i]); labels[i] = NULL; }
    }
    labels_count = 0;
}
