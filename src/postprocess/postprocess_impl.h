#ifndef _POSTPROCESS_IMPL_H_
#define _POSTPROCESS_IMPL_H_

#include "../postprocess.h"
#include "../rknnprocess.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

int postprocess_yolov8(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path);
void deinit_postprocess_yolov8(void);

int postprocess_classification(struct _RknnProcess* rknn_process,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path);
void deinit_postprocess_classification(void);

int postprocess_retinaface(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path);
void deinit_postprocess_retinaface(void);

int postprocess_yolov6(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path);
void deinit_postprocess_yolov6(void);

#ifdef __cplusplus
}
#endif

#endif /* _POSTPROCESS_IMPL_H_ */
