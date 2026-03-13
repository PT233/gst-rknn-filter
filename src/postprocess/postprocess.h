#ifndef _POSTPROCESS_DISPATCH_H_
#define _POSTPROCESS_DISPATCH_H_

#include "postprocess_common.h"
#include "rknnprocess.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/* Unified postprocess dispatcher - returns 0 on success */
int postprocess_dispatch(struct _RknnProcess* rknn_process,
    void* orig_img,
    float box_conf_threshold, float nms_threshold,
    int show_fps, double current_fps, int do_inference,
    detect_result_group_t* group);

void deinit_postprocess_all(void);

#ifdef __cplusplus
}
#endif

#endif /* _POSTPROCESS_DISPATCH_H_ */
