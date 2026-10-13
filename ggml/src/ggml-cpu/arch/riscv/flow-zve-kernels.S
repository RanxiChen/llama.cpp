/* Flow single-zve32f GEMV kernels: one activation row against groups of 128
 * repacked weight rows.
 *
 * Cost model (R07b, hardware 4cbc473): each vector unit (ALU, MAC, FMA/conv,
 * stream) runs one instruction at a time and only accepts the next after all
 * write-backs are acked (~12.6 cycles for an m1 ALU op, ~15 for vwmacc), and
 * the window has 16 entries. So the kernels minimise instruction count (large
 * vl/LMUL, 128 rows per vwmacc) and double-buffer the weight registers (sets
 * A/B) so the next vle8 and nibble split can run while MAC works on the
 * current set.
 *
 * Only fast-path instructions: unit-stride vle/vse, vsll/vsra .vi, vmv.v.i,
 * vwmul/vwmacc, vfwcvt/vfcvt, vfmacc. No reductions, slides, vrgather,
 * vmv.x.s, vfmul, vwadd or vsext (element engine, slow FP lane or a vector
 * drain), no strided/indexed accesses (Serial path).
 *
 * Needs VLEN >= 512 (e8/m4 with vl=256 = two 128-row weight columns).
 * Leaf functions; only caller-saved scalar registers are used. */
    .option arch, +zve32f
    .text

/* uint64_t flow_zve_vlenb(void) */
    .globl flow_zve_vlenb
    .type flow_zve_vlenb, @function
flow_zve_vlenb:
    csrr a0, vlenb
    ret
    .size flow_zve_vlenb, . - flow_zve_vlenb

/* Q4 load+split of two columns (j, j+1) into a set: \ld <- 256 B, then
 * \lo = sign-extended low nibbles (elements j, j+1), \ld = high nibbles
 * (elements j+16, j+17) in place. Nibbles were xor'd with 8 at repack.
 * Runs at e8/m4, vl=256 (t2). */
.macro Q4LOAD ld, lo
    vsetvli zero, t2, e8, m4, ta, ma
    vle8.v \ld, (a0)
    addi a0, a0, 256
    vsll.vi \lo, \ld, 4
    vsra.vi \lo, \lo, 4
    vsra.vi \ld, \ld, 4
.endm

/* Q4 MACs for one set at e8/m2, vl=128 (t0); t3 points at qa[j]. Each
 * vwmacc covers 128 rows of one element: v20..v23 = 128 x i16. */
.macro Q4MAC ld, lo, ld1, lo1
    vsetvli zero, t0, e8, m2, ta, ma
    lb t1, 0(t3)
    lb a6, 16(t3)
    vwmacc.vx v20, t1, \lo
    vwmacc.vx v20, a6, \ld
    lb t1, 1(t3)
    lb a6, 17(t3)
    vwmacc.vx v20, t1, \lo1
    vwmacc.vx v20, a6, \ld1
    addi t3, t3, 2
.endm

/* Block epilogue shared by both kernels: \conv (m8 f32 sums of 128 rows)
 * times da*dw accumulated into v0..v7. dw (128 f32) follows the quants. */
.macro EPILOGUE conv
    vsetvli zero, t0, e32, m8, ta, ma
    vle32.v v24, (a0)
    addi a0, a0, 512
    flw ft0, 0(t4)
    addi t4, t4, 4
    vmv.v.i v16, 0
    vfmacc.vf v16, ft0, v24         /* da*dw without vfmul (slow lane) */
    vfmacc.vv v0, v16, \conv
.endm

/* void flow_zve_gemv_q4_0(const uint8_t *w, const int8_t *qa, const float *da,
 *                         float *y, size_t nblocks, size_t ngroups)
 * w:  ngroups x nblocks x 2560 B: qs[j*128 + r] (16 x 128 B), then 128 f32 dw
 * qa: nblocks x 32 int8, da: nblocks f32, y: ngroups x 128 f32
 * Registers: v0-7 f32 row sums; set A v8-11/v12-15; set B v24-27/v28-31;
 * v20-23 i16 block sums (|sum| <= 32*8*127 < 2^15); v16-23, v24-31 and
 * v8-15 are reused by the epilogue once both sets are consumed. */
    .globl flow_zve_gemv_q4_0
    .type flow_zve_gemv_q4_0, @function
flow_zve_gemv_q4_0:
    li t0, 128
    li t2, 256
1:
    vsetvli zero, t0, e32, m8, ta, ma
    vmv.v.i v0, 0
    mv t3, a1
    mv t4, a2
    mv t5, a4
2:
    vsetvli zero, t0, e16, m4, ta, ma
    vmv.v.i v20, 0
    li t6, 4
    Q4LOAD v8, v12
3:
    Q4LOAD v24, v28
    Q4MAC v8, v12, v10, v14
    addi t6, t6, -1
    beqz t6, 4f
    Q4LOAD v8, v12
    Q4MAC v24, v28, v26, v30
    j 3b
4:
    Q4MAC v24, v28, v26, v30
    addi t3, t3, 16
    vsetvli zero, t0, e16, m4, ta, ma
    vfwcvt.f.x.v v8, v20
    EPILOGUE v8
    addi t5, t5, -1
    bnez t5, 2b
    vsetvli zero, t0, e32, m8, ta, ma
    vse32.v v0, (a3)
    addi a3, a3, 512
    addi a5, a5, -1
    bnez a5, 1b
    ret
    .size flow_zve_gemv_q4_0, . - flow_zve_gemv_q4_0

/* Q8 load of two columns (j, j+1) into a set, e8/m4, vl=256. */
.macro Q8LOAD ld
    vsetvli zero, t2, e8, m4, ta, ma
    vle8.v \ld, (a0)
    addi a0, a0, 256
.endm

/* Q8 MACs for one set: two int8 products fit i16 (2*127*127 < 2^15); the
 * pair is widened into the i32 sums by vwmacc with multiplier 1 (a7),
 * because vwadd/vsext are not on the fast path. */
.macro Q8MAC ld, ld1, part
    vsetvli zero, t0, e8, m2, ta, ma
    lb t1, 0(t3)
    lb a6, 1(t3)
    vwmul.vx \part, \ld, t1
    vwmacc.vx \part, a6, \ld1
    vsetvli zero, t0, e16, m4, ta, ma
    vwmacc.vx v8, a7, \part
    addi t3, t3, 2
.endm

/* void flow_zve_gemv_q8_0(const uint8_t *w, const int8_t *qa, const float *da,
 *                         float *y, size_t nblocks, size_t ngroups)
 * w: ngroups x nblocks x 4608 B: qs[j*128 + r] (32 x 128 B), then 128 f32 dw.
 * Registers: v0-7 f32 row sums; v8-15 i32 block sums, converted in place;
 * set A v16-19 + part v20-23; set B v24-27 + part v28-31. */
    .globl flow_zve_gemv_q8_0
    .type flow_zve_gemv_q8_0, @function
flow_zve_gemv_q8_0:
    li t0, 128
    li t2, 256
    li a7, 1
1:
    vsetvli zero, t0, e32, m8, ta, ma
    vmv.v.i v0, 0
    mv t3, a1
    mv t4, a2
    mv t5, a4
2:
    vsetvli zero, t0, e32, m8, ta, ma
    vmv.v.i v8, 0
    li t6, 8
    Q8LOAD v16
3:
    Q8LOAD v24
    Q8MAC v16, v18, v20
    addi t6, t6, -1
    beqz t6, 4f
    Q8LOAD v16
    Q8MAC v24, v26, v28
    j 3b
4:
    Q8MAC v24, v26, v28
    vsetvli zero, t0, e32, m8, ta, ma
    vfcvt.f.x.v v8, v8              /* |sum| <= 32*127*127 < 2^24: exact */
    EPILOGUE v8
    addi t5, t5, -1
    bnez t5, 2b
    vsetvli zero, t0, e32, m8, ta, ma
    vse32.v v0, (a3)
    addi a3, a3, 512
    addi a5, a5, -1
    bnez a5, 1b
    ret
    .size flow_zve_gemv_q8_0, . - flow_zve_gemv_q8_0
