// Postprocess dispatcher - routes to model-specific postprocess based on model_type

#include "../postprocess.h"
#include "../rknnprocess.h"
#include "postprocess_impl.h"
#include <vector>
#include <cstring>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int postprocess_dispatch(struct _RknnProcess* rknn_process,
    void* orig_img,
    float box_conf_threshold, float nms_threshold,
    int show_fps, double current_fps, int do_inference,
    detect_result_group_t* group)
{
    int ret = 0;

    if (do_inference) {
        rknn_outputs_release(rknn_process->ctx, rknn_process->io_num.n_output, rknn_process->outputs);
        rknn_inputs_set(rknn_process->ctx, rknn_process->io_num.n_input, rknn_process->inputs);
        ret = rknn_run(rknn_process->ctx, NULL);
        ret = rknn_outputs_get(rknn_process->ctx, rknn_process->io_num.n_output, rknn_process->outputs, NULL);
        if (ret != 0) return ret;
    }

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (uint32_t i = 0; i < rknn_process->io_num.n_output; i++) {
        out_scales.push_back(rknn_process->output_attrs[i].scale);
        out_zps.push_back(rknn_process->output_attrs[i].zp);
    }

    memset(group, 0, sizeof(detect_result_group_t));

    switch (rknn_process->model_type) {
    case RKNN_MODEL_YOLOV5:
    case RKNN_MODEL_YOLOV7:
        if (rknn_process->io_num.n_output >= 3) {
            post_process(
                (int8_t*)rknn_process->outputs[0].buf,
                (int8_t*)rknn_process->outputs[1].buf,
                (int8_t*)rknn_process->outputs[2].buf,
                rknn_process->model_height, rknn_process->model_width,
                box_conf_threshold, nms_threshold,
                rknn_process->pads, rknn_process->scale_w, rknn_process->scale_h,
                out_zps, out_scales, group, rknn_process->label_path);
        }
        break;
    case RKNN_MODEL_YOLOV6:
        if (rknn_process->io_num.n_output >= 9) {
            postprocess_yolov6(rknn_process, box_conf_threshold, nms_threshold,
                out_zps, out_scales, group, rknn_process->label_path);
        }
        break;
    case RKNN_MODEL_YOLOV8:
    case RKNN_MODEL_YOLOV8_OBB:
    case RKNN_MODEL_YOLOV10:
    case RKNN_MODEL_YOLO11:
    case RKNN_MODEL_PPYOLOE:
        postprocess_yolov8(rknn_process, box_conf_threshold, nms_threshold,
            out_zps, out_scales, group, rknn_process->label_path);
        break;
    case RKNN_MODEL_RETINAFACE:
        postprocess_retinaface(rknn_process, box_conf_threshold, nms_threshold,
            out_zps, out_scales, group, rknn_process->label_path);
        break;
    case RKNN_MODEL_YOLOV8_POSE:
    case RKNN_MODEL_DEEPLABV3:
    case RKNN_MODEL_YOLOV5_SEG:
    case RKNN_MODEL_YOLOV8_SEG:
    case RKNN_MODEL_PPSEG:
    case RKNN_MODEL_LPRNET:
    case RKNN_MODEL_PPOCR_DET:
    case RKNN_MODEL_PPOCR_REC:
    case RKNN_MODEL_PASSTHROUGH:
    default:
        group->count = 0;
        break;
    }
    return 0;
}

void deinit_postprocess_all(void)
{
    deinitPostProcess();
    deinit_postprocess_yolov8();
    deinit_postprocess_yolov6();
    deinit_postprocess_retinaface();
}

#ifdef __cplusplus
}
#endif
