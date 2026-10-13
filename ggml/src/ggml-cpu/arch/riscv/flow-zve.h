#pragma once

// Flow single-zve32f repack path: Q4_0 / Q8_0 weights in groups of 128 rows,
// f32 scales, GEMV kernels in flow-zve-kernels.S. Enabled by GGML_FLOW_ZVE32F.

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#define FLOW_ZVE_ROWS 128

#ifdef __cplusplus
extern "C" {
#endif

// 1 when the kernels may run: hwprobe reports Zve32f (or FLOW_ZVE=force),
// vlenb >= 64, and FLOW_ZVE is not "0". Decided once and logged.
int flow_zve_available(void);

// Shape/type check used both for buffer sizing and for traits selection.
int flow_zve_supports(const struct ggml_tensor * t);

size_t flow_zve_group_block_bytes(enum ggml_type type);
size_t flow_zve_repacked_bytes(const struct ggml_tensor * t);
size_t flow_zve_act_row_bytes(int64_t k);

int  flow_zve_repack(struct ggml_tensor * t, const void * data, size_t data_size);
void flow_zve_quantize_act(const float * x, void * row, int64_t k);
void flow_zve_gemv(enum ggml_type type, const void * w, const void * act_row, float * y, int64_t k, int64_t ngroups);

#ifdef __cplusplus
}
#endif
