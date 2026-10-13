// Flow single-zve32f repack path. Weights are repacked at load time into
// groups of FLOW_ZVE_ROWS rows; per 32-element block a group stores
// qs[j*ROWS + r] for every column j, then ROWS f32 scales (no Zvfh, so the
// fp16 scales are widened here). Q4_0 nibbles are xor'd with 8 so the kernel
// can sign-extend them with shifts. Activations use the same values as
// quantize_row_q8_0_ref (int8 + fp16-rounded scale), stored as int8[k]
// followed by f32[k/32].

#include "ggml-quants.h"
#include "ggml-impl.h"

#include "../../quants.h"
#include "flow-zve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>
#endif

#ifndef RISCV_HWPROBE_KEY_IMA_EXT_0
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#endif
#ifndef RISCV_HWPROBE_EXT_ZVE32F
#define RISCV_HWPROBE_EXT_ZVE32F (1ULL << 38)
#endif

#define FLOW_ZVE_Q4_BLOCK (QK4_0 / 2 * FLOW_ZVE_ROWS + 4 * FLOW_ZVE_ROWS)
#define FLOW_ZVE_Q8_BLOCK (QK8_0 * FLOW_ZVE_ROWS + 4 * FLOW_ZVE_ROWS)

uint64_t flow_zve_vlenb(void);
void flow_zve_gemv_q4_0(const uint8_t * w, const int8_t * qa, const float * da, float * y, size_t nblocks, size_t ngroups);
void flow_zve_gemv_q8_0(const uint8_t * w, const int8_t * qa, const float * da, float * y, size_t nblocks, size_t ngroups);

static int flow_zve_selftest(void);

static int flow_zve_hwprobe_zve32f(void) {
#if defined(__linux__) && defined(__NR_riscv_hwprobe)
    struct riscv_hwprobe probe = { .key = RISCV_HWPROBE_KEY_IMA_EXT_0, .value = 0 };
    if (syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0) != 0) {
        return 0;
    }
    return (probe.value & RISCV_HWPROBE_EXT_ZVE32F) != 0;
#else
    return 0;
#endif
}

int flow_zve_available(void) {
    static int state = -1;
    if (state >= 0) {
        return state;
    }

    // FLOW_ZVE=0 keeps the stock scalar path in the same binary (A/B runs);
    // FLOW_ZVE=force skips hwprobe (e.g. qemu-user without hwprobe bits).
    const char * env   = getenv("FLOW_ZVE");
    const char * why   = NULL;
    uint64_t     vlenb = 0;
    if (env && strcmp(env, "0") == 0) {
        why = "FLOW_ZVE=0";
    } else if (!(env && strcmp(env, "force") == 0) && !flow_zve_hwprobe_zve32f()) {
        why = "hwprobe reports no zve32f";
    } else if ((vlenb = flow_zve_vlenb()) < 64) {
        why = "vlenb < 64";
    }

    state = why == NULL;
    if (state) {
        GGML_LOG_INFO("flow-zve: Q4_0/Q8_0 %d-row kernels enabled (vlenb=%llu)\n", FLOW_ZVE_ROWS, (unsigned long long) vlenb);
        if (getenv("FLOW_ZVE_SELFTEST") && !flow_zve_selftest()) {
            GGML_ABORT("flow-zve: self-test failed");
        }
    } else {
        GGML_LOG_INFO("flow-zve: disabled (%s)\n", why);
    }
    return state;
}

int flow_zve_supports(const struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q8_0) {
        return 0;
    }
    if (t->ne[0] <= 0 || t->ne[0] % QK8_0 != 0 || t->ne[1] % FLOW_ZVE_ROWS != 0 || t->ne[2] != 1 || t->ne[3] != 1) {
        return 0;
    }
    return flow_zve_available();
}

size_t flow_zve_group_block_bytes(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? FLOW_ZVE_Q4_BLOCK : FLOW_ZVE_Q8_BLOCK;
}

size_t flow_zve_repacked_bytes(const struct ggml_tensor * t) {
    return (size_t) (t->ne[1] / FLOW_ZVE_ROWS) * (size_t) (t->ne[0] / QK8_0) * flow_zve_group_block_bytes(t->type);
}

size_t flow_zve_act_row_bytes(int64_t k) {
    return (size_t) k + (size_t) (k / QK8_0) * sizeof(float);
}

static void flow_zve_repack_q4_0(uint8_t * dst, const block_q4_0 * src, int64_t nrows, int64_t nb) {
    for (int64_t g = 0; g < nrows / FLOW_ZVE_ROWS; ++g) {
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t * out = dst + (g * nb + b) * FLOW_ZVE_Q4_BLOCK;
            float *   dw  = (float *) (out + QK4_0 / 2 * FLOW_ZVE_ROWS);
            for (int r = 0; r < FLOW_ZVE_ROWS; ++r) {
                const block_q4_0 * x = src + (g * FLOW_ZVE_ROWS + r) * nb + b;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    out[j * FLOW_ZVE_ROWS + r] = x->qs[j] ^ 0x88;
                }
                dw[r] = GGML_FP16_TO_FP32(x->d);
            }
        }
    }
}

static void flow_zve_repack_q8_0(uint8_t * dst, const block_q8_0 * src, int64_t nrows, int64_t nb) {
    for (int64_t g = 0; g < nrows / FLOW_ZVE_ROWS; ++g) {
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t * out = dst + (g * nb + b) * FLOW_ZVE_Q8_BLOCK;
            float *   dw  = (float *) (out + QK8_0 * FLOW_ZVE_ROWS);
            for (int r = 0; r < FLOW_ZVE_ROWS; ++r) {
                const block_q8_0 * x = src + (g * FLOW_ZVE_ROWS + r) * nb + b;
                for (int j = 0; j < QK8_0; ++j) {
                    // The kernel sums two products in i16; -128 * -128 * 2 would wrap.
                    // quantize_row_q8_0_ref never emits -128.
                    GGML_ASSERT(x->qs[j] != -128);
                    out[j * FLOW_ZVE_ROWS + r] = (uint8_t) x->qs[j];
                }
                dw[r] = GGML_FP16_TO_FP32(x->d);
            }
        }
    }
}

int flow_zve_repack(struct ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(flow_zve_supports(t));
    GGML_ASSERT(data_size == ggml_nbytes(t));
    const int64_t nb = t->ne[0] / QK8_0;
    if (t->type == GGML_TYPE_Q4_0) {
        flow_zve_repack_q4_0((uint8_t *) t->data, (const block_q4_0 *) data, t->ne[1], nb);
    } else {
        flow_zve_repack_q8_0((uint8_t *) t->data, (const block_q8_0 *) data, t->ne[1], nb);
    }
    return 0;
}

void flow_zve_quantize_act(const float * x, void * row, int64_t k) {
    int8_t * qa = (int8_t *) row;
    float *  da = (float *) ((uint8_t *) row + k);
    for (int64_t i = 0; i < k / QK8_0; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = MAX(amax, fabsf(x[i * QK8_0 + j]));
        }
        // Same arithmetic as quantize_row_q8_0_ref, including the fp16 scale.
        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        da[i] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(d));
        for (int j = 0; j < QK8_0; ++j) {
            qa[i * QK8_0 + j] = roundf(x[i * QK8_0 + j] * id);
        }
    }
}

void flow_zve_gemv(enum ggml_type type, const void * w, const void * act_row, float * y, int64_t k, int64_t ngroups) {
    const int8_t * qa = (const int8_t *) act_row;
    const float *  da = (const float *) ((const uint8_t *) act_row + k);
    if (type == GGML_TYPE_Q4_0) {
        flow_zve_gemv_q4_0((const uint8_t *) w, qa, da, y, (size_t) (k / QK8_0), (size_t) ngroups);
    } else {
        flow_zve_gemv_q8_0((const uint8_t *) w, qa, da, y, (size_t) (k / QK8_0), (size_t) ngroups);
    }
}

// Self-test (FLOW_ZVE_SELFTEST=1): repack + kernel against ggml's generic
// vec_dot on the same quantized data. Checks that the activation bytes match
// quantize_row_q8_0_ref exactly and that every output is within 1e-5 of the
// sum of |block terms| (a wrong row/column mapping is off by O(1) of that).

static uint32_t flow_zve_rng = 0x2545f491u;

static float flow_zve_rand(void) {
    flow_zve_rng ^= flow_zve_rng << 13;
    flow_zve_rng ^= flow_zve_rng >> 17;
    flow_zve_rng ^= flow_zve_rng << 5;
    return (float) (flow_zve_rng >> 8) / (float) (1u << 23) - 1.0f;
}

static int flow_zve_selftest_one(enum ggml_type type, int64_t k, int64_t n) {
    const int64_t nb      = k / QK8_0;
    const size_t  wq_row  = ggml_row_size(type, k);
    float *       wf      = malloc(sizeof(float) * k);
    uint8_t *     wq      = malloc(wq_row * n);
    float *       x       = malloc(sizeof(float) * k);
    block_q8_0 *  xq      = malloc(sizeof(block_q8_0) * nb);
    struct ggml_tensor t  = { 0 };
    t.type  = type;
    t.ne[0] = k; t.ne[1] = n; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = wq_row; t.nb[2] = wq_row * n; t.nb[3] = wq_row * n;
    uint8_t * packed = malloc(flow_zve_repacked_bytes(&t));
    uint8_t * act    = malloc(flow_zve_act_row_bytes(k));
    float *   y      = malloc(sizeof(float) * n);
    float *   ref    = malloc(sizeof(float) * n);
    int       ok     = 1;

    for (int64_t r = 0; r < n; ++r) {
        const float scale = 0.01f + 0.2f * (float) (r % 7);
        for (int64_t i = 0; i < k; ++i) {
            wf[i] = scale * flow_zve_rand();
        }
        if (type == GGML_TYPE_Q4_0) {
            quantize_row_q4_0_ref(wf, (block_q4_0 *) (wq + r * wq_row), k);
        } else {
            quantize_row_q8_0_ref(wf, (block_q8_0 *) (wq + r * wq_row), k);
        }
    }
    for (int64_t i = 0; i < k; ++i) {
        // Block 0 all zero exercises d = 0; other blocks vary in magnitude.
        x[i] = i < QK8_0 ? 0.0f : (1.0f + (float) ((i / QK8_0) % 5)) * flow_zve_rand();
    }
    quantize_row_q8_0_ref(x, xq, k);

    t.data = packed;
    flow_zve_repack(&t, wq, wq_row * n);
    flow_zve_quantize_act(x, act, k);
    for (int64_t b = 0; b < nb && ok; ++b) {
        const float da = ((const float *) (act + k))[b];
        ok = memcmp(act + b * QK8_0, xq[b].qs, QK8_0) == 0 && da == GGML_FP16_TO_FP32(xq[b].d);
    }
    if (!ok) {
        GGML_LOG_ERROR("flow-zve: self-test %s: activation quantization differs from quantize_row_q8_0_ref\n", ggml_type_name(type));
    }

    flow_zve_gemv(type, packed, act, y, k, n / FLOW_ZVE_ROWS);

    float max_err = 0.0f;
    for (int64_t r = 0; r < n && ok; ++r) {
        const uint8_t * row = wq + r * wq_row;
        float           mag = 0.0f;
        for (int64_t b = 0; b < nb; ++b) {
            int   sumabs = 0;
            float dw;
            if (type == GGML_TYPE_Q4_0) {
                const block_q4_0 * w = (const block_q4_0 *) row + b;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    sumabs += abs(((w->qs[j] & 15) - 8) * xq[b].qs[j]) + abs(((w->qs[j] >> 4) - 8) * xq[b].qs[j + 16]);
                }
                dw = GGML_FP16_TO_FP32(w->d);
            } else {
                const block_q8_0 * w = (const block_q8_0 *) row + b;
                for (int j = 0; j < QK8_0; ++j) {
                    sumabs += abs(w->qs[j] * xq[b].qs[j]);
                }
                dw = GGML_FP16_TO_FP32(w->d);
            }
            mag += (float) sumabs * fabsf(dw * GGML_FP16_TO_FP32(xq[b].d));
        }
        if (type == GGML_TYPE_Q4_0) {
            ggml_vec_dot_q4_0_q8_0_generic((int) k, &ref[r], 0, row, 0, xq, 0, 1);
        } else {
            ggml_vec_dot_q8_0_q8_0_generic((int) k, &ref[r], 0, row, 0, xq, 0, 1);
        }
        const float err = fabsf(y[r] - ref[r]);
        max_err = MAX(max_err, mag > 0.0f ? err / mag : err);
        if (!(err <= 1e-5f * mag)) {
            GGML_LOG_ERROR("flow-zve: self-test %s k=%lld n=%lld row %lld: got %g, ref %g (mag %g)\n", ggml_type_name(type),
                           (long long) k, (long long) n, (long long) r, (double) y[r], (double) ref[r], (double) mag);
            ok = 0;
        }
    }
    printf("flow-zve: self-test %s k=%lld n=%lld: %s (max err/mag %.3g)\n", ggml_type_name(type), (long long) k,
           (long long) n, ok ? "PASS" : "FAIL", (double) max_err);
    fflush(stdout);

    free(wf); free(wq); free(x); free(xq); free(packed); free(act); free(y); free(ref);
    return ok;
}

static int flow_zve_selftest(void) {
    int ok = 1;
    ok &= flow_zve_selftest_one(GGML_TYPE_Q4_0, 896, 256);
    ok &= flow_zve_selftest_one(GGML_TYPE_Q4_0, 4864, 128);
    ok &= flow_zve_selftest_one(GGML_TYPE_Q8_0, 896, 256);
    ok &= flow_zve_selftest_one(GGML_TYPE_Q8_0, 64, 128);
    return ok;
}
