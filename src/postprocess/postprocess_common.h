#ifndef _POSTPROCESS_COMMON_H_
#define _POSTPROCESS_COMMON_H_

#include "../postprocess.h"

/* Common dequant helper */
inline static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

#endif /* _POSTPROCESS_COMMON_H_ */
