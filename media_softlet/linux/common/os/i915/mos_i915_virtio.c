/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Virtio-GPU i915 native context support for the iHD media driver.
 *
 * Self-contained vdrm client + i915 ccmd translation layer adapted from
 * mesa's src/virtio/vdrm/ and src/intel/dev/virtio/i915_virtio_ccmd.c,
 * and matched to the host side implemented in virglrenderer's
 * src/drm/i915/i915_ccmd.c.
 *
 * Limitations of the host side protocol that are worked around here:
 *
 *  - GEM objects must be created as virtgpu blobs (RESOURCE_CREATE_BLOB
 *    with a tunneled GEM_CREATE ccmd); mmap uses VIRTGPU_MAP.
 *  - Relocations are not translated by the host, so the bufmgr must run
 *    in softpin mode (submissions with final guest addresses);
 *    mos_bufmgr.c forces softpin when running on virtio.
 *  - I915_CONTEXT_PARAM_ENGINES cannot be set after context creation and
 *    must not carry load-balance/bond extensions.  Context creation is
 *    therefore deferred until the first use, engine parameters are
 *    translated into a plain static engine map, and exec flags are
 *    rewritten from legacy ring selectors into engine-map indexes.
 *  - The host reserves 8 fence-context groups per vdrm connection, so at
 *    most 8 concurrently live GEM contexts are supported; contexts with
 *    identical creation parameters share one host context.
 *  - GEM_WAIT/GEM_BUSY are emulated with DRM_IOCTL_VIRTGPU_WAIT.
 *  - In/out fences (I915_EXEC_FENCE_IN/OUT/SUBMIT) are emulated with the
 *    virtgpu execbuffer fence_fd in/out mechanism.
 *  - GEM_USERPTR, GEM_FLINK and GEM_OPEN are not supported by the host.
 *
 * This file is compiled as C++ (like all .c files in iHD) via
 * set_source_files_properties(... LANGUAGE "CXX") in cmake.
 */

#include "mos_i915_virtio.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <xf86drm.h>
#include <libdrm/virtgpu_drm.h>

/* i915 uapi headers, the same copies mos_bufmgr.c uses */
#include "drm.h"
#include "i915_drm.h"

#define MOS_I915_VIRTIO_LOG(fmt, ...) \
    fprintf(stderr, "intel-media-driver-virtio: " fmt "\n", ##__VA_ARGS__)

/* ======================================================================
 * Helpers
 * ====================================================================== */

static inline uint32_t
virtio_align(uint32_t v, uint32_t a)
{
    return (v + a - 1) & ~(a - 1);
}

#ifndef MIN2
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* DRM ioctls vary in size depending on the uapi header version used by
 * the caller; compare them ignoring the size bits.
 */
#define IOC_MASKED(IOC) ((IOC) & ~IOCSIZE_MASK)

/* All ioctl handlers follow the libdrm drmIoctl() convention: return 0
 * on success, -1 on failure with errno set.
 */
#define VIRTIO_ERR(e) do { errno = (e); return -1; } while (0)

/* ======================================================================
 * vdrm protocol structures (from mesa's virtio/virtio-gpu/drm_hw.h)
 * ====================================================================== */

#ifndef VIRTGPU_DRM_CONTEXT_I915
#define VIRTGPU_DRM_CONTEXT_I915 3
#endif

struct virgl_renderer_capset_drm {
    uint32_t wire_format_version;
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patchlevel;
    uint32_t context_type;
    uint32_t pad;
    union {
        struct {
            uint32_t pci_bus;
            uint32_t pci_dev;
            uint32_t pci_func;
            uint32_t pci_revision_id;
            uint32_t pci_domain;
            uint32_t pci_device_id;
        } intel;
    } u;
};

struct vdrm_shmem {
    uint32_t seqno;
    uint32_t rsp_mem_offset;
};

struct vdrm_ccmd_req {
    uint32_t cmd;
    uint32_t len;
    uint32_t seqno;
    uint32_t rsp_off;
};

struct vdrm_ccmd_rsp {
    uint32_t len;
};

/* ======================================================================
 * i915 virtio protocol structures (from virglrenderer's
 * src/drm/i915/i915_proto.h).  Must be kept in sync with the host.
 * ====================================================================== */

enum i915_ccmd {
    I915_CCMD_IOCTL_SIMPLE = 1,
    I915_CCMD_GETPARAM,
    I915_CCMD_QUERYPARAM,
    I915_CCMD_GEM_CREATE,
    I915_CCMD_GEM_CREATE_EXT,
    I915_CCMD_GEM_CONTEXT_CREATE,
    I915_CCMD_GEM_EXECBUFFER2,
    I915_CCMD_GEM_SET_MMAP_MODE,
    I915_CCMD_GEM_BUSY,
    I915_CCMD_GEM_EXECBUFFER2_V2,
};

struct i915_shmem {
    struct vdrm_shmem base;
    uint64_t banned_ctx_mask;
};

struct i915_ccmd_ioctl_simple_req {
    struct vdrm_ccmd_req hdr;
    uint32_t cmd;
    uint32_t pad;
    uint8_t payload[];
};

struct i915_ccmd_ioctl_simple_rsp {
    struct vdrm_ccmd_rsp hdr;
    int32_t ret;
    uint8_t payload[];
};

struct i915_ccmd_getparam_req {
    struct vdrm_ccmd_req hdr;
    uint32_t param;
    uint32_t value;
};

struct i915_ccmd_getparam_rsp {
    struct vdrm_ccmd_rsp hdr;
    uint32_t pad;
    int32_t ret;
    uint32_t value;
};

struct i915_ccmd_queryparam_req {
    struct vdrm_ccmd_req hdr;
    uint32_t query_id;
    uint32_t length;
    uint32_t flags;
    uint32_t pad;
};

struct i915_ccmd_queryparam_rsp {
    struct vdrm_ccmd_rsp hdr;
    uint32_t pad;
    int32_t ret;
    int32_t length;
    uint8_t payload[];
};

struct i915_ccmd_gem_context_create_req {
    struct vdrm_ccmd_req hdr;
    uint32_t flags;
    uint32_t params_size;
    uint8_t payload[];
};

struct i915_ccmd_gem_context_create_rsp {
    struct vdrm_ccmd_rsp hdr;
    uint32_t pad;
    int32_t ret;
    uint32_t ctx_id;
};

struct i915_ccmd_gem_create_req {
    struct vdrm_ccmd_req hdr;
    uint64_t size;
    uint32_t blob_id;
    uint32_t pad;
};

struct i915_ccmd_gem_create_ext_req {
    struct vdrm_ccmd_req hdr;
    uint64_t size;
    uint32_t blob_id;
    uint32_t gem_flags;
    uint32_t ext_size;
    uint32_t pad;
    uint8_t payload[];
};

struct i915_ccmd_gem_execbuffer2_req {
    struct vdrm_ccmd_req hdr;
    uint64_t flags;
    uint64_t context_id;
    uint32_t buffer_count;
    uint32_t batch_start_offset;
    uint32_t batch_len;
    uint32_t relocs_count;
    uint8_t payload[];
};

struct i915_ccmd_gem_execbuffer2_rsp {
    struct vdrm_ccmd_rsp hdr;
    int32_t ret;
};

struct i915_ccmd_gem_set_mmap_mode_req {
    struct vdrm_ccmd_req hdr;
    uint32_t res_id;
    uint32_t flags;
};

struct i915_ccmd_gem_busy_req {
    struct vdrm_ccmd_req hdr;
    uint32_t res_id;
    uint32_t pad;
};

struct i915_ccmd_gem_busy_rsp {
    struct vdrm_ccmd_rsp hdr;
    uint32_t pad;
    int32_t ret;
    int32_t busy;
};

/* ======================================================================
 * vdrm client state
 * ====================================================================== */

#define VDRM_SHMEM_SZ  0x4000
#define VDRM_REQBUF_SZ 0x4000

struct vdrm_device {
    int fd; /* vdrm protocol fd (may be reopened) */

    struct vdrm_shmem *shmem;
    uint8_t *rsp_mem;
    uint32_t rsp_mem_len;
    uint32_t next_rsp_off;

    uint32_t shmem_handle;

    uint32_t next_seqno;
    uint32_t next_blob_id;

    pthread_mutex_t rsp_lock;
    pthread_mutex_t eb_lock;

    uint32_t reqbuf_len;
    uint32_t reqbuf_cnt;
    uint8_t reqbuf[VDRM_REQBUF_SZ];

    struct virgl_renderer_capset_drm caps;
    bool supports_cross_device;
};

/* Single-instance design: only one active vdrm connection at a time.
 * This matches the media driver's usage model (one bufmgr per device
 * fd).  Multi-device support would require per-fd state tracking.
 */
static struct vdrm_device *g_vdrm = NULL;
static int g_vdrm_fd = -1;
/* fd that was already probed and found not to be a virtio-gpu i915 fd */
static int g_non_virtio_fd = -1;
static pthread_mutex_t g_vdrm_global_lock = PTHREAD_MUTEX_INITIALIZER;

/* ======================================================================
 * Handle -> blob_id mapping
 *
 * The virglrenderer host tracks GEM objects by their blob_id (set via
 * drm_context_object_set_blob_id in the ccmd handlers), not by the
 * kernel-assigned resource ID returned by VIRTGPU_RESOURCE_INFO.  We
 * therefore record the blob_id assigned at BO creation time so that
 * execbuffer can send blob_ids in the ccmd payload, allowing the host
 * to translate handles correctly.
 * ====================================================================== */

struct blob_map_entry {
    uint32_t handle;
    uint32_t blob_id;
    struct blob_map_entry *next;
};

static struct blob_map_entry *g_blob_map = NULL;
static pthread_mutex_t g_blob_map_lock = PTHREAD_MUTEX_INITIALIZER;

static void
blob_map_add(uint32_t handle, uint32_t blob_id)
{
    struct blob_map_entry *entry = calloc(1, sizeof(*entry));
    if (!entry)
        return;
    entry->handle = handle;
    entry->blob_id = blob_id;
    pthread_mutex_lock(&g_blob_map_lock);
    entry->next = g_blob_map;
    g_blob_map = entry;
    pthread_mutex_unlock(&g_blob_map_lock);
}

static uint32_t
blob_map_lookup(uint32_t handle)
{
    uint32_t result = 0;
    pthread_mutex_lock(&g_blob_map_lock);
    struct blob_map_entry *entry = g_blob_map;
    while (entry) {
        if (entry->handle == handle) {
            result = entry->blob_id;
            break;
        }
        entry = entry->next;
    }
    pthread_mutex_unlock(&g_blob_map_lock);
    return result;
}

static void
blob_map_remove(uint32_t handle)
{
    pthread_mutex_lock(&g_blob_map_lock);
    struct blob_map_entry **p = &g_blob_map;
    while (*p) {
        if ((*p)->handle == handle) {
            struct blob_map_entry *tmp = *p;
            *p = (*p)->next;
            free(tmp);
            break;
        }
        p = &(*p)->next;
    }
    pthread_mutex_unlock(&g_blob_map_lock);
}

/* ======================================================================
 * GEM context bookkeeping
 *
 * The host rejects I915_CONTEXT_PARAM_ENGINES set after context creation
 * and rejects engine parameters that carry load-balance/bond extensions.
 * Host contexts are therefore created lazily (at first use), with a
 * static engine map derived from the driver's engine parameters, and
 * driver contexts sharing the same creation parameters share the host
 * context.  The i915 native context protocol reserves 8 fence-context
 * groups per vdrm connection, so at most 8 host contexts can exist.
 * ====================================================================== */

#define I915_VIRTIO_MAX_HOST_CTX 8

struct i915_virtio_engine {
    uint16_t engine_class;
    uint16_t engine_instance;
};

struct i915_virtio_param {
    uint32_t param;
    uint32_t size;
    uint64_t value; /* scalar value, used when size == 0 */
    uint8_t *data;
};

/* Driver visible context */
struct i915_virtio_ctx {
    uint32_t guest_id;
    uint32_t create_flags;
    struct i915_virtio_param *params;
    uint32_t num_params;
    int host_idx; /* index into g_host_ctxs, -1 if not created yet */
};

/* Host side context pool entry */
struct i915_virtio_host_ctx {
    bool in_use;
    uint32_t host_ctx_id;
    uint32_t refcount;
    uint32_t create_flags;
    struct i915_virtio_param *params;
    uint32_t num_params;
    /* static engine map (empty when the context has no engine map) */
    struct i915_virtio_engine *map;
    uint32_t map_count;
};

static struct i915_virtio_ctx **g_ctxs = NULL;
static uint32_t g_ctx_count = 0;
static uint32_t g_ctx_capacity = 0;
static uint32_t g_next_guest_ctx_id = 0;
static struct i915_virtio_host_ctx g_host_ctxs[I915_VIRTIO_MAX_HOST_CTX];

static void
free_param(struct i915_virtio_param *p)
{
    free(p->data);
    memset(p, 0, sizeof(*p));
}

static void
free_params(struct i915_virtio_param *params, uint32_t num_params)
{
    for (uint32_t i = 0; i < num_params; i++)
        free_param(&params[i]);
    free(params);
}

static void
reset_context_state(void)
{
    for (uint32_t i = 0; i < g_ctx_count; i++) {
        if (g_ctxs[i]) {
            free_params(g_ctxs[i]->params, g_ctxs[i]->num_params);
            free(g_ctxs[i]);
            g_ctxs[i] = NULL;
        }
    }
    free(g_ctxs);
    g_ctxs = NULL;
    g_ctx_count = 0;
    g_ctx_capacity = 0;
    g_next_guest_ctx_id = 0;

    for (uint32_t i = 0; i < I915_VIRTIO_MAX_HOST_CTX; i++) {
        struct i915_virtio_host_ctx *h = &g_host_ctxs[i];
        if (!h->in_use)
            continue;
        free_params(h->params, h->num_params);
        free(h->map);
        memset(h, 0, sizeof(*h));
    }
}

/* Translate the driver's I915_CONTEXT_PARAM_ENGINES value (which carries
 * a load-balance/bond/parallel-submit extension) into a plain static
 * engine map that the host accepts (extensions must be zero).
 *
 * On success returns a malloc'd engines payload (struct
 * i915_context_param_engines layout: __u64 extensions followed by engine
 * instances) of *out_size bytes plus the extracted map.
 */
static uint8_t *
translate_engines_param(const uint8_t *data, uint32_t size,
                        uint32_t *out_size,
                        struct i915_virtio_engine **out_map,
                        uint32_t *out_map_count)
{
    struct i915_virtio_engine engines[64];
    uint32_t count = 0;
    uint8_t *out;
    uint64_t extensions;

    *out_size = 0;
    *out_map = NULL;
    *out_map_count = 0;

    if (size < sizeof(uint64_t))
        return NULL;

    memcpy(&extensions, data, sizeof(extensions));

    if (extensions == 0) {
        /* Already a plain engine map: pass through as-is. */
        uint32_t map_bytes = size - (uint32_t)sizeof(uint64_t);
        if (map_bytes % sizeof(struct i915_engine_class_instance) != 0)
            return NULL;

        *out_map_count = map_bytes / sizeof(struct i915_engine_class_instance);
        if (*out_map_count > 0) {
            *out_map = (struct i915_virtio_engine *)malloc(map_bytes);
            if (!*out_map) {
                *out_map_count = 0;
                return NULL;
            }
            memcpy(*out_map, data + sizeof(uint64_t), map_bytes);
        }

        out = (uint8_t *)malloc(size);
        if (!out) {
            free(*out_map);
            *out_map = NULL;
            *out_map_count = 0;
            return NULL;
        }
        memcpy(out, data, size);
        *out_size = size;
        return out;
    }

    const struct i915_user_extension *ext =
        (const struct i915_user_extension *)(uintptr_t)extensions;

    while (ext) {
        switch (ext->name) {
        case I915_CONTEXT_ENGINES_EXT_LOAD_BALANCE: {
            const struct i915_context_engines_load_balance *lb =
                (const struct i915_context_engines_load_balance *)ext;
            for (uint16_t i = 0; i < lb->num_siblings; i++) {
                if (count >= ARRAY_SIZE(engines))
                    return NULL;
                engines[count].engine_class = lb->engines[i].engine_class;
                engines[count].engine_instance = lb->engines[i].engine_instance;
                count++;
            }
            break;
        }
        case I915_CONTEXT_ENGINES_EXT_BOND: {
            const struct i915_context_engines_bond *bond =
                (const struct i915_context_engines_bond *)ext;
            for (uint16_t i = 0; i < bond->num_bonds; i++) {
                if (count >= ARRAY_SIZE(engines))
                    return NULL;
                engines[count].engine_class = bond->engines[i].engine_class;
                engines[count].engine_instance = bond->engines[i].engine_instance;
                count++;
            }
            break;
        }
        case I915_CONTEXT_ENGINES_EXT_PARALLEL_SUBMIT: {
            const struct i915_context_engines_parallel_submit *ps =
                (const struct i915_context_engines_parallel_submit *)ext;
            uint32_t n = (uint32_t)ps->width * ps->num_siblings;
            for (uint32_t i = 0; i < n; i++) {
                if (count >= ARRAY_SIZE(engines))
                    return NULL;
                engines[count].engine_class = ps->engines[i].engine_class;
                engines[count].engine_instance = ps->engines[i].engine_instance;
                count++;
            }
            break;
        }
        default:
            return NULL;
        }

        ext = (const struct i915_user_extension *)(uintptr_t)ext->next_extension;
    }

    if (count == 0)
        return NULL;

    *out_size = (uint32_t)sizeof(uint64_t) +
                count * (uint32_t)sizeof(struct i915_engine_class_instance);

    out = (uint8_t *)malloc(*out_size);
    if (!out) {
        *out_size = 0;
        return NULL;
    }

    memset(out, 0, sizeof(uint64_t)); /* extensions == 0 */
    for (uint32_t i = 0; i < count; i++) {
        struct i915_engine_class_instance ci;
        ci.engine_class = engines[i].engine_class;
        ci.engine_instance = engines[i].engine_instance;
        memcpy(out + sizeof(uint64_t) + i * sizeof(ci), &ci, sizeof(ci));
    }

    *out_map = (struct i915_virtio_engine *)malloc(count * sizeof(**out_map));
    if (!*out_map) {
        free(out);
        *out_size = 0;
        return NULL;
    }
    memcpy(*out_map, engines, count * sizeof(**out_map));
    *out_map_count = count;

    return out;
}

/* Context parameters the host accepts at context creation time. */
static bool
param_supported_at_create(uint32_t param)
{
    switch (param) {
    case I915_CONTEXT_PARAM_BAN_PERIOD:
    case I915_CONTEXT_PARAM_GTT_SIZE:
    case I915_CONTEXT_PARAM_NO_ERROR_CAPTURE:
    case I915_CONTEXT_PARAM_BANNABLE:
    case I915_CONTEXT_PARAM_PRIORITY:
    case I915_CONTEXT_PARAM_SSEU:
    case I915_CONTEXT_PARAM_RECOVERABLE:
    case I915_CONTEXT_PARAM_VM:
    case I915_CONTEXT_PARAM_ENGINES:
    case I915_CONTEXT_PARAM_PERSISTENCE:
    case I915_CONTEXT_PARAM_PROTECTED_CONTENT:
        return true;
    default:
        return false;
    }
}

/* Context parameters the host accepts via the plain setparam ioctl
 * after the context has been created.
 */
static bool
param_supported_post_create(uint32_t param)
{
    switch (param) {
    case I915_CONTEXT_PARAM_PRIORITY:
    case I915_CONTEXT_PARAM_RECOVERABLE:
    case I915_CONTEXT_PARAM_GTT_SIZE:
    case I915_CONTEXT_PARAM_VM:
        return true;
    default:
        return false;
    }
}

static int
append_param(struct i915_virtio_param **params, uint32_t *num_params,
             uint32_t param, uint32_t size, uint64_t value, const uint8_t *data)
{
    if (*num_params == 0) {
        *params = (struct i915_virtio_param *)calloc(1, sizeof(**params));
        if (!*params)
            return -ENOMEM;
    } else {
        struct i915_virtio_param *p = (struct i915_virtio_param *)realloc(
            *params, (*num_params + 1) * sizeof(**params));
        if (!p)
            return -ENOMEM;
        *params = p;
    }

    struct i915_virtio_param *entry = &(*params)[*num_params];
    memset(entry, 0, sizeof(*entry));
    entry->param = param;
    entry->size = size;
    entry->value = value;
    if (size > 0) {
        entry->data = (uint8_t *)malloc(size);
        if (!entry->data)
            return -ENOMEM;
        memcpy(entry->data, data, size);
    }
    (*num_params)++;
    return 0;
}

static struct i915_virtio_ctx *
ctx_lookup(uint32_t guest_id)
{
    for (uint32_t i = 0; i < g_ctx_count; i++) {
        if (g_ctxs[i] && g_ctxs[i]->guest_id == guest_id)
            return g_ctxs[i];
    }
    return NULL;
}

static struct i915_virtio_ctx *
ctx_create(uint32_t create_flags)
{
    uint32_t slot = g_ctx_count;

    for (uint32_t i = 0; i < g_ctx_count; i++) {
        if (!g_ctxs[i]) {
            slot = i;
            break;
        }
    }

    if (slot == g_ctx_count) {
        if (g_ctx_count == g_ctx_capacity) {
            uint32_t new_cap = g_ctx_capacity ? g_ctx_capacity * 2 : 16;
            struct i915_virtio_ctx **new_ctxs =
                (struct i915_virtio_ctx **)realloc(
                    g_ctxs, new_cap * sizeof(*new_ctxs));
            if (!new_ctxs)
                return NULL;
            memset(new_ctxs + g_ctx_capacity, 0,
                   (new_cap - g_ctx_capacity) * sizeof(*new_ctxs));
            g_ctxs = new_ctxs;
            g_ctx_capacity = new_cap;
        }
        g_ctx_count++;
    }

    struct i915_virtio_ctx *ctx =
        (struct i915_virtio_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        if (slot == g_ctx_count)
            g_ctx_count--;
        return NULL;
    }

    ctx->guest_id = ++g_next_guest_ctx_id;
    ctx->create_flags = create_flags;
    ctx->host_idx = -1;
    g_ctxs[slot] = ctx;
    return ctx;
}

static void
ctx_destroy(struct i915_virtio_ctx *ctx)
{
    if (ctx->host_idx >= 0) {
        struct i915_virtio_host_ctx *h = &g_host_ctxs[ctx->host_idx];
        if (h->refcount > 0)
            h->refcount--;
        ctx->host_idx = -1;
    }

    for (uint32_t i = 0; i < g_ctx_count; i++) {
        if (g_ctxs[i] == ctx) {
            free_params(ctx->params, ctx->num_params);
            free(ctx);
            g_ctxs[i] = NULL;
            return;
        }
    }
}

static bool
params_equal(const struct i915_virtio_param *a, uint32_t num_a,
             const struct i915_virtio_param *b, uint32_t num_b)
{
    if (num_a != num_b)
        return false;
    for (uint32_t i = 0; i < num_a; i++) {
        if (a[i].param != b[i].param || a[i].size != b[i].size)
            return false;
        if (a[i].size == 0) {
            if (a[i].value != b[i].value)
                return false;
        } else if (memcmp(a[i].data, b[i].data, a[i].size) != 0) {
            return false;
        }
    }
    return true;
}

/* Forward declarations used by the deferred context creation below. */
static int i915_virtio_simple_ioctl(unsigned long cmd, void *_req,
                                    bool patch_handle);
static void *vdrm_alloc_rsp(struct vdrm_device *vdev,
                            struct vdrm_ccmd_req *req, uint32_t sz);
static int vdrm_send_req(struct vdrm_device *vdev, struct vdrm_ccmd_req *req,
                         bool sync);

/* Send the GEM_CONTEXT_CREATE ccmd (or plain create) for a host
 * context.
 */
static int
host_ctx_send_create(struct i915_virtio_host_ctx *h)
{
    struct vdrm_device *vdev = g_vdrm;
    uint32_t params_size = 0;

    for (uint32_t i = 0; i < h->num_params; i++)
        params_size += (uint32_t)sizeof(struct drm_i915_gem_context_create_ext_setparam) +
                       h->params[i].size;

    if (params_size == 0) {
        /* Plain context create, no extensions. */
        struct drm_i915_gem_context_create create;
        memset(&create, 0, sizeof(create));

        int err = i915_virtio_simple_ioctl(DRM_IOCTL_I915_GEM_CONTEXT_CREATE,
                                           &create, false);
        if (err)
            return err;

        h->host_ctx_id = create.ctx_id;
        if (h->host_ctx_id == 0 || h->host_ctx_id > I915_VIRTIO_MAX_HOST_CTX) {
            MOS_I915_VIRTIO_LOG("unexpected host ctx_id %u", h->host_ctx_id);
            return EINVAL;
        }
        return 0;
    }

    uint32_t req_len = (uint32_t)sizeof(struct i915_ccmd_gem_context_create_req) +
                       params_size;
    uint8_t *buf = (uint8_t *)malloc(req_len);
    if (!buf)
        return ENOMEM;

    struct i915_ccmd_gem_context_create_req *req =
        (struct i915_ccmd_gem_context_create_req *)buf;
    memset(req, 0, sizeof(*req));
    req->hdr.cmd = I915_CCMD_GEM_CONTEXT_CREATE;
    req->hdr.len = req_len;
    req->flags = h->create_flags | I915_CONTEXT_CREATE_FLAGS_USE_EXTENSIONS;
    req->params_size = params_size;

    uint8_t *payload = req->payload;
    for (uint32_t i = 0; i < h->num_params; i++) {
        struct drm_i915_gem_context_create_ext_setparam sp;
        memset(&sp, 0, sizeof(sp));
        sp.base.name = I915_CONTEXT_CREATE_EXT_SETPARAM;
        sp.param.param = h->params[i].param;
        sp.param.size = h->params[i].size;
        sp.param.value = h->params[i].size ? 0 : h->params[i].value;
        memcpy(payload, &sp, sizeof(sp));
        payload += sizeof(sp);
        if (h->params[i].size > 0) {
            memcpy(payload, h->params[i].data, h->params[i].size);
            payload += h->params[i].size;
        }
    }

    struct i915_ccmd_gem_context_create_rsp *rsp =
        (struct i915_ccmd_gem_context_create_rsp *)
        vdrm_alloc_rsp(vdev, &req->hdr, sizeof(*rsp));

    int err = vdrm_send_req(vdev, &req->hdr, true);
    if (err) {
        free(buf);
        return errno ? errno : EIO;
    }

    int ret = 0;
    if (rsp->ret) {
        ret = rsp->ret;
    } else {
        h->host_ctx_id = rsp->ctx_id;
        if (h->host_ctx_id == 0 || h->host_ctx_id > I915_VIRTIO_MAX_HOST_CTX) {
            MOS_I915_VIRTIO_LOG("unexpected host ctx_id %u", h->host_ctx_id);
            ret = EINVAL;
        }
    }

    free(buf);
    return ret;
}

/* Find or create the host context matching the given creation
 * parameters.  On success takes ownership of @p map and returns the
 * pool index; on failure returns a negative errno.
 */
static int
host_ctx_get(uint32_t create_flags,
             const struct i915_virtio_param *params, uint32_t num_params,
             struct i915_virtio_engine *map, uint32_t map_count)
{
    for (uint32_t i = 0; i < I915_VIRTIO_MAX_HOST_CTX; i++) {
        struct i915_virtio_host_ctx *h = &g_host_ctxs[i];
        if (!h->in_use)
            continue;
        if (h->create_flags == create_flags &&
            params_equal(h->params, h->num_params, params, num_params)) {
            h->refcount++;
            free(map); /* existing entry keeps its own map */
            return (int)i;
        }
    }

    for (uint32_t i = 0; i < I915_VIRTIO_MAX_HOST_CTX; i++) {
        struct i915_virtio_host_ctx *h = &g_host_ctxs[i];
        if (h->in_use)
            continue;

        memset(h, 0, sizeof(*h));
        h->in_use = true;
        h->create_flags = create_flags;
        h->refcount = 1;
        h->map = map;
        h->map_count = map_count;

        if (num_params > 0) {
            h->params = (struct i915_virtio_param *)calloc(
                num_params, sizeof(*h->params));
            if (!h->params) {
                free(map);
                memset(h, 0, sizeof(*h));
                return -ENOMEM;
            }
            for (uint32_t j = 0; j < num_params; j++) {
                h->params[j].param = params[j].param;
                h->params[j].size = params[j].size;
                h->params[j].value = params[j].value;
                if (params[j].size > 0) {
                    h->params[j].data = (uint8_t *)malloc(params[j].size);
                    if (!h->params[j].data) {
                        free_params(h->params, j);
                        free(map);
                        memset(h, 0, sizeof(*h));
                        return -ENOMEM;
                    }
                    memcpy(h->params[j].data, params[j].data, params[j].size);
                }
            }
            h->num_params = num_params;
        }

        int ret = host_ctx_send_create(h);
        if (ret != 0) {
            free_params(h->params, h->num_params);
            free(h->map);
            memset(h, 0, sizeof(*h));
            return -ret;
        }
        return (int)i;
    }

    MOS_I915_VIRTIO_LOG("too many live GEM contexts (max %d)",
                        I915_VIRTIO_MAX_HOST_CTX);
    free(map);
    return -EINVAL;
}

/* Extract the static engine map from the recorded (already translated)
 * ENGINES param of a driver context.  Returns a malloc'd map that the
 * caller must free (or hand over to host_ctx_get()).
 */
static struct i915_virtio_engine *
ctx_engine_map(struct i915_virtio_ctx *ctx, uint32_t *map_count)
{
    *map_count = 0;

    for (uint32_t i = 0; i < ctx->num_params; i++) {
        if (ctx->params[i].param != I915_CONTEXT_PARAM_ENGINES ||
            ctx->params[i].size == 0)
            continue;

        /* The stored payload is already a plain engine map
         * (extensions == 0), so this simply extracts it.
         */
        uint32_t size = 0;
        struct i915_virtio_engine *map = NULL;
        uint8_t *payload = translate_engines_param(ctx->params[i].data,
                                                   ctx->params[i].size,
                                                   &size, &map, map_count);
        free(payload);
        return map;
    }
    return NULL;
}

/* Make sure the driver context has a host context. */
static int
ctx_ensure_host(struct i915_virtio_ctx *ctx)
{
    if (ctx->host_idx >= 0)
        return 0;

    uint32_t map_count = 0;
    struct i915_virtio_engine *map = ctx_engine_map(ctx, &map_count);

    int idx = host_ctx_get(ctx->create_flags, ctx->params, ctx->num_params,
                           map, map_count);
    if (idx < 0)
        return idx;

    ctx->host_idx = idx;
    return 0;
}

/* Rewrite legacy exec flags into an index into the context engine map.
 * With a user engine map the I915_EXEC_RING_MASK field is an index into
 * the map, and the BSD ring selectors are not honored.
 */
static uint64_t
translate_exec_flags(struct i915_virtio_host_ctx *h, uint64_t flags)
{
    if (h->map_count == 0)
        return flags;

    uint32_t ring = (uint32_t)(flags & I915_EXEC_RING_MASK);
    uint32_t bsd_ring = (uint32_t)((flags & I915_EXEC_BSD_MASK) >>
                                   I915_EXEC_BSD_SHIFT);
    uint16_t want_class;
    uint16_t want_instance;
    uint32_t map_idx = 0;
    bool found = false;

    if (ring == I915_EXEC_DEFAULT) {
        map_idx = 0;
        found = true;
    } else {
        switch (ring) {
        case I915_EXEC_RENDER:
            want_class = I915_ENGINE_CLASS_RENDER;
            want_instance = 0;
            break;
        case I915_EXEC_BSD:
            want_class = I915_ENGINE_CLASS_VIDEO;
            want_instance = bsd_ring > 0 ? (uint16_t)(bsd_ring - 1) : 0;
            break;
        case I915_EXEC_BLT:
            want_class = I915_ENGINE_CLASS_COPY;
            want_instance = 0;
            break;
        case I915_EXEC_VEBOX:
            want_class = I915_ENGINE_CLASS_VIDEO_ENHANCE;
            want_instance = 0;
            break;
        default:
            return flags;
        }

        for (uint32_t i = 0; i < h->map_count; i++) {
            if (h->map[i].engine_class == want_class &&
                h->map[i].engine_instance == want_instance) {
                map_idx = i;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        MOS_I915_VIRTIO_LOG("engine class %u not in context map, using map[0]",
                            want_class);
    }

    uint64_t new_flags = flags & ~((uint64_t)I915_EXEC_RING_MASK |
                                   (uint64_t)I915_EXEC_BSD_MASK);
    new_flags |= map_idx;
    return new_flags;
}

/* ======================================================================
 * Low level virtgpu helpers
 * ====================================================================== */

static void
virtgpu_gem_close(int fd, uint32_t handle)
{
    struct drm_gem_close c;
    memset(&c, 0, sizeof(c));
    c.handle = handle;
    blob_map_remove(handle);
    drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &c);
}

static int
virtgpu_map_handle(int fd, uint32_t handle, size_t size, void **out, void *placed)
{
    struct drm_virtgpu_map req;
    memset(&req, 0, sizeof(req));
    req.handle = handle;

    int ret = drmIoctl(fd, DRM_IOCTL_VIRTGPU_MAP, &req);
    if (ret) {
        MOS_I915_VIRTIO_LOG("VIRTGPU_MAP failed: %s", strerror(errno));
        return ret;
    }

    *out = mmap(placed, size, PROT_READ | PROT_WRITE,
                MAP_SHARED | (placed ? MAP_FIXED : 0),
                fd, req.offset);
    if (*out == MAP_FAILED) {
        MOS_I915_VIRTIO_LOG("mmap failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

/* ======================================================================
 * vdrm client implementation
 * ====================================================================== */

static int
vdrm_get_capset(int fd, struct virgl_renderer_capset_drm *caps)
{
    struct drm_virtgpu_get_caps args;
    memset(&args, 0, sizeof(args));
    args.cap_set_id = VIRTGPU_DRM_CAPSET_DRM;
    args.cap_set_ver = 0;
    args.addr = (uintptr_t)caps;
    args.size = sizeof(*caps);

    memset(caps, 0, sizeof(*caps));
    return drmIoctl(fd, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
}

static int
vdrm_set_context(int fd)
{
    struct drm_virtgpu_context_set_param params[] = {
        { VIRTGPU_CONTEXT_PARAM_CAPSET_ID, VIRTGPU_DRM_CAPSET_DRM },
        { VIRTGPU_CONTEXT_PARAM_NUM_RINGS, 64 },
    };
    struct drm_virtgpu_context_init args;
    memset(&args, 0, sizeof(args));
    args.num_params = ARRAY_SIZE(params);
    args.ctx_set_params = (uintptr_t)params;

    return drmIoctl(fd, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &args);
}

static int
vdrm_init_shmem(struct vdrm_device *vdev)
{
    struct drm_virtgpu_resource_create_blob blob;
    memset(&blob, 0, sizeof(blob));
    blob.blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
    blob.blob_flags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
    blob.size = VDRM_SHMEM_SZ;
    blob.blob_id = 0;

    int ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &blob);
    if (ret) {
        MOS_I915_VIRTIO_LOG("vdrm shmem alloc failed: %s", strerror(errno));
        return ret;
    }

    vdev->shmem_handle = blob.bo_handle;

    ret = virtgpu_map_handle(vdev->fd, vdev->shmem_handle,
                             blob.size, (void **)&vdev->shmem, NULL);
    if (ret) {
        virtgpu_gem_close(vdev->fd, vdev->shmem_handle);
        vdev->shmem_handle = 0;
        return ret;
    }

    uint32_t offset = vdev->shmem->rsp_mem_offset;
    vdev->rsp_mem_len = VDRM_SHMEM_SZ - offset;
    vdev->rsp_mem = &((uint8_t *)vdev->shmem)[offset];

    return 0;
}

static struct vdrm_device *
vdrm_connect(int fd)
{
    struct virgl_renderer_capset_drm caps;
    int ret;

    ret = vdrm_get_capset(fd, &caps);
    if (ret)
        return NULL;

    if (caps.context_type != VIRTGPU_DRM_CONTEXT_I915)
        return NULL;

    ret = vdrm_set_context(fd);
    if (ret && errno == EEXIST) {
        /* Context already exists on this fd (e.g. mesa).  Reopen to get
         * our own context -- needed for shmem mmap.
         */
        char path[64];
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        int own_fd = open(path, O_RDWR | O_CLOEXEC);
        if (own_fd < 0)
            return NULL;
        ret = vdrm_set_context(own_fd);
        if (ret) {
            close(own_fd);
            return NULL;
        }
        fd = own_fd;
    } else if (ret) {
        return NULL;
    }

    struct vdrm_device *vdev =
        (struct vdrm_device *)calloc(1, sizeof(*vdev));
    if (!vdev)
        return NULL;

    vdev->fd = fd;
    pthread_mutex_init(&vdev->rsp_lock, NULL);
    pthread_mutex_init(&vdev->eb_lock, NULL);

    ret = vdrm_init_shmem(vdev);
    if (ret) {
        free(vdev);
        return NULL;
    }

    vdev->caps = caps;
    return vdev;
}

static void
vdrm_disconnect(struct vdrm_device *vdev)
{
    if (!vdev)
        return;

    /* Flush outstanding requests before tearing down. */
    pthread_mutex_lock(&vdev->eb_lock);
    if (vdev->reqbuf_len > 0) {
        struct drm_virtgpu_execbuffer eb;
        memset(&eb, 0, sizeof(eb));
        eb.flags = VIRTGPU_EXECBUF_RING_IDX;
        eb.size = vdev->reqbuf_len;
        eb.command = (uintptr_t)vdev->reqbuf;
        drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &eb);
        vdev->reqbuf_len = 0;
        vdev->reqbuf_cnt = 0;
    }
    pthread_mutex_unlock(&vdev->eb_lock);

    if (vdev->shmem)
        munmap(vdev->shmem, VDRM_SHMEM_SZ);
    if (vdev->shmem_handle)
        virtgpu_gem_close(vdev->fd, vdev->shmem_handle);

    pthread_mutex_destroy(&vdev->rsp_lock);
    pthread_mutex_destroy(&vdev->eb_lock);

    if (vdev->fd != g_vdrm_fd)
        close(vdev->fd);

    free(vdev);
}

/* ---- response allocation ---- */

static void *
vdrm_alloc_rsp(struct vdrm_device *vdev, struct vdrm_ccmd_req *req, uint32_t sz)
{
    uint32_t off;

    pthread_mutex_lock(&vdev->rsp_lock);

    sz = virtio_align(sz, 8);

    if ((vdev->next_rsp_off + sz) >= vdev->rsp_mem_len)
        vdev->next_rsp_off = 0;

    off = vdev->next_rsp_off;
    vdev->next_rsp_off += sz;

    pthread_mutex_unlock(&vdev->rsp_lock);

    req->rsp_off = off;

    struct vdrm_ccmd_rsp *rsp = (struct vdrm_ccmd_rsp *)&vdev->rsp_mem[off];
    rsp->len = sz;

    return rsp;
}

/* ---- fence helpers ---- */

static bool
fence_before(uint32_t a, uint32_t b)
{
    return (int32_t)(a - b) < 0;
}

static void
vdrm_host_sync(struct vdrm_device *vdev, const struct vdrm_ccmd_req *req)
{
    while (fence_before(vdev->shmem->seqno, req->seqno))
        sched_yield();
}

/* ---- flush / execbuf ---- */

static int
vdrm_flush_locked(struct vdrm_device *vdev, int *out_fence_fd)
{
    if (!vdev->reqbuf_len)
        return 0;

    struct drm_virtgpu_execbuffer eb;
    memset(&eb, 0, sizeof(eb));
    eb.flags = VIRTGPU_EXECBUF_RING_IDX;
    eb.size = vdev->reqbuf_len;
    eb.command = (uintptr_t)vdev->reqbuf;

    if (out_fence_fd)
        eb.flags |= VIRTGPU_EXECBUF_FENCE_FD_OUT;

    int ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &eb);
    if (ret) {
        MOS_I915_VIRTIO_LOG("VIRTGPU_EXECBUFFER flush failed: %s",
                            strerror(errno));
        return ret;
    }

    vdev->reqbuf_len = 0;
    vdev->reqbuf_cnt = 0;

    if (out_fence_fd)
        *out_fence_fd = eb.fence_fd;

    return 0;
}

static int
vdrm_enqueue_req(struct vdrm_device *vdev, struct vdrm_ccmd_req *req)
{
    /* Caller holds eb_lock */
    req->seqno = ++vdev->next_seqno;

    if ((vdev->reqbuf_len + req->len) > sizeof(vdev->reqbuf)) {
        int ret = vdrm_flush_locked(vdev, NULL);
        if (ret)
            return ret;
    }

    memcpy(&vdev->reqbuf[vdev->reqbuf_len], req, req->len);
    vdev->reqbuf_len += req->len;
    vdev->reqbuf_cnt++;

    return 0;
}

/**
 * Send a ccmd request. If @p sync is true, flush and wait for the
 * host to process the command (blocking).
 */
static int
vdrm_send_req(struct vdrm_device *vdev, struct vdrm_ccmd_req *req, bool sync)
{
    int fence_fd = -1;
    int ret = 0;

    pthread_mutex_lock(&vdev->eb_lock);
    ret = vdrm_enqueue_req(vdev, req);

    if (ret || !sync)
        goto unlock;

    ret = vdrm_flush_locked(vdev, &fence_fd);

unlock:
    pthread_mutex_unlock(&vdev->eb_lock);

    if (ret)
        return ret;

    if (sync) {
        /* Wait for the fence, then spin until host writes seqno. */
        if (fence_fd >= 0) {
            struct pollfd pfd;
            memset(&pfd, 0, sizeof(pfd));
            pfd.fd = fence_fd;
            pfd.events = POLLIN;
            while (poll(&pfd, 1, -1) == -1 && errno == EINTR)
                ;
            close(fence_fd);
        }
        vdrm_host_sync(vdev, req);
    }

    return 0;
}

/**
 * Create a GEM object on the host via RESOURCE_CREATE_BLOB + ccmd.
 * Returns the guest GEM handle, or 0 on failure.
 */
static uint32_t
vdrm_bo_create(struct vdrm_device *vdev, size_t size, uint32_t blob_flags,
               uint64_t blob_id, struct vdrm_ccmd_req *req)
{
    pthread_mutex_lock(&vdev->eb_lock);

    /* Flush buffered cmds so host sees them before the blob create. */
    vdrm_flush_locked(vdev, NULL);

    req->seqno = ++vdev->next_seqno;

    struct drm_virtgpu_resource_create_blob args;
    memset(&args, 0, sizeof(args));
    args.blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
    args.blob_flags = blob_flags;
    args.size = size;
    args.cmd_size = req->len;
    args.cmd = (uintptr_t)req;
    args.blob_id = blob_id;

    int ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args);

    pthread_mutex_unlock(&vdev->eb_lock);

    if (ret) {
        MOS_I915_VIRTIO_LOG("vdrm bo_create failed: %s", strerror(errno));
        return 0;
    }

    blob_map_add(args.bo_handle, (uint32_t)blob_id);

    return args.bo_handle;
}

/**
 * Close a BO -- flush first so detach_resource doesn't race.
 */
static void
vdrm_bo_close(struct vdrm_device *vdev, uint32_t handle)
{
    if (vdev->reqbuf_len) {
        pthread_mutex_lock(&vdev->eb_lock);
        vdrm_flush_locked(vdev, NULL);
        pthread_mutex_unlock(&vdev->eb_lock);
    }
    virtgpu_gem_close(vdev->fd, handle);
}

/**
 * Get the virtgpu resource id for a GEM handle.
 */
static uint32_t
vdrm_handle_to_res_id(struct vdrm_device *vdev, uint32_t handle)
{
    struct drm_virtgpu_resource_info info;
    memset(&info, 0, sizeof(info));
    info.bo_handle = handle;

    int ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_RESOURCE_INFO, &info);
    if (ret) {
        MOS_I915_VIRTIO_LOG("RESOURCE_INFO failed: %s", strerror(errno));
        return 0;
    }
    return info.res_handle;
}

/* ======================================================================
 * i915 ioctl handlers
 * ====================================================================== */

/* Forward declaration: used by the deferred context creation. */
static int
i915_virtio_simple_ioctl(unsigned long cmd, void *_req, bool patch_handle);

/* I915_CCMD_IOCTL_SIMPLE: pass a fixed size ioctl through to the host.
 * @p patch_handle translates the first payload dword (a GEM handle) to
 * a virtgpu resource id.
 */
static int
i915_virtio_simple_ioctl(unsigned long cmd, void *_req, bool patch_handle)
{
    struct vdrm_device *vdev = g_vdrm;
    uint32_t payload_len = (uint32_t)_IOC_SIZE(cmd);
    uint32_t req_len = (uint32_t)sizeof(struct i915_ccmd_ioctl_simple_req) +
                       payload_len;
    uint32_t rsp_len = (uint32_t)sizeof(struct i915_ccmd_ioctl_simple_rsp);
    bool sync = !!(cmd & IOC_OUT);
    int ret = 0;

    if (payload_len == 0)
        return EINVAL;

    if (cmd & IOC_OUT)
        rsp_len += payload_len;

    uint8_t *buf = (uint8_t *)malloc(req_len);
    if (!buf)
        return ENOMEM;

    struct i915_ccmd_ioctl_simple_req *req =
        (struct i915_ccmd_ioctl_simple_req *)buf;
    memset(req, 0, sizeof(*req));
    req->hdr.cmd = I915_CCMD_IOCTL_SIMPLE;
    req->hdr.len = req_len;
    req->cmd = (uint32_t)cmd;
    memcpy(req->payload, _req, payload_len);

    if (patch_handle) {
        uint32_t *handle = (uint32_t *)req->payload;
        *handle = vdrm_handle_to_res_id(vdev, *handle);
    }

    struct i915_ccmd_ioctl_simple_rsp *rsp =
        (struct i915_ccmd_ioctl_simple_rsp *)
        vdrm_alloc_rsp(vdev, &req->hdr, rsp_len);

    int err = vdrm_send_req(vdev, &req->hdr, sync);
    if (err) {
        free(buf);
        return errno ? errno : EIO;
    }

    if (sync) {
        if (cmd & IOC_OUT)
            memcpy(_req, rsp->payload, payload_len);
        if (rsp->ret)
            ret = rsp->ret;
    }

    free(buf);
    return ret;
}

static int
i915_virtio_queryparam(struct drm_i915_query *query)
{
    struct vdrm_device *vdev = g_vdrm;
    struct drm_i915_query_item *item =
        (struct drm_i915_query_item *)(uintptr_t)query->items_ptr;

    if (query->num_items != 1) {
        MOS_I915_VIRTIO_LOG("unsupported number of query items");
        return EINVAL;
    }

    struct i915_ccmd_queryparam_req req;
    memset(&req, 0, sizeof(req));
    req.hdr.cmd = I915_CCMD_QUERYPARAM;
    req.hdr.len = sizeof(req);
    req.query_id = item->query_id;
    req.length = item->length;
    req.flags = item->flags;

    uint32_t rsp_len = (uint32_t)sizeof(struct i915_ccmd_queryparam_rsp) +
                       item->length;
    struct i915_ccmd_queryparam_rsp *rsp =
        (struct i915_ccmd_queryparam_rsp *)
        vdrm_alloc_rsp(vdev, &req.hdr, rsp_len);

    int err = vdrm_send_req(vdev, &req.hdr, true);
    if (err)
        return errno ? errno : EIO;

    if (item->data_ptr && rsp->length > 0)
        memcpy((void *)(uintptr_t)item->data_ptr, rsp->payload, rsp->length);

    item->length = rsp->length;

    return rsp->ret;
}

static int
i915_virtio_getparam(struct drm_i915_getparam *gp)
{
    struct vdrm_device *vdev = g_vdrm;
    struct i915_ccmd_getparam_req req;
    memset(&req, 0, sizeof(req));
    req.hdr.cmd = I915_CCMD_GETPARAM;
    req.hdr.len = sizeof(req);
    req.param = gp->param;

    struct i915_ccmd_getparam_rsp *rsp =
        (struct i915_ccmd_getparam_rsp *)
        vdrm_alloc_rsp(vdev, &req.hdr, sizeof(*rsp));

    int err = vdrm_send_req(vdev, &req.hdr, true);
    if (err)
        return errno ? errno : EIO;

    *gp->value = rsp->value;

    return rsp->ret;
}

/* Flatten a GEM_CREATE_EXT extension chain into the ccmd payload. */
static int
gem_create_ext_payload(const struct drm_i915_gem_create_ext *create,
                       uint8_t **out_payload, uint32_t *out_size)
{
    const struct i915_user_extension *ext =
        (const struct i915_user_extension *)(uintptr_t)create->extensions;
    uint32_t size = 0;
    uint8_t *payload;
    uint8_t *ptr;

    *out_payload = NULL;
    *out_size = 0;

    while (ext) {
        switch (ext->name) {
        case I915_GEM_CREATE_EXT_MEMORY_REGIONS: {
            const struct drm_i915_gem_create_ext_memory_regions *mr =
                (const struct drm_i915_gem_create_ext_memory_regions *)ext;
            size += (uint32_t)sizeof(*mr);
            size += (uint32_t)sizeof(struct drm_i915_gem_memory_class_instance) *
                    mr->num_regions;
            break;
        }
        case I915_GEM_CREATE_EXT_PROTECTED_CONTENT:
            size += (uint32_t)sizeof(struct drm_i915_gem_create_ext_protected_content);
            break;
        case I915_GEM_CREATE_EXT_SET_PAT:
            size += (uint32_t)sizeof(struct drm_i915_gem_create_ext_set_pat);
            break;
        default:
            MOS_I915_VIRTIO_LOG("unsupported gem_create_ext extension %u",
                                ext->name);
            return EINVAL;
        }
        ext = (const struct i915_user_extension *)(uintptr_t)ext->next_extension;
    }

    if (size == 0)
        return 0;

    payload = (uint8_t *)calloc(1, size);
    if (!payload)
        return ENOMEM;

    ext = (const struct i915_user_extension *)(uintptr_t)create->extensions;
    ptr = payload;
    while (ext) {
        switch (ext->name) {
        case I915_GEM_CREATE_EXT_MEMORY_REGIONS: {
            const struct drm_i915_gem_create_ext_memory_regions *mr =
                (const struct drm_i915_gem_create_ext_memory_regions *)ext;
            memcpy(ptr, mr, sizeof(*mr));
            ptr += sizeof(*mr);
            if (mr->num_regions > 0) {
                const struct drm_i915_gem_memory_class_instance *instances =
                    (const struct drm_i915_gem_memory_class_instance *)
                    (uintptr_t)mr->regions;
                memcpy(ptr, instances,
                       sizeof(*instances) * mr->num_regions);
                ptr += sizeof(*instances) * mr->num_regions;
            }
            break;
        }
        case I915_GEM_CREATE_EXT_PROTECTED_CONTENT:
            memcpy(ptr, ext,
                   sizeof(struct drm_i915_gem_create_ext_protected_content));
            ptr += sizeof(struct drm_i915_gem_create_ext_protected_content);
            break;
        case I915_GEM_CREATE_EXT_SET_PAT:
            memcpy(ptr, ext, sizeof(struct drm_i915_gem_create_ext_set_pat));
            ptr += sizeof(struct drm_i915_gem_create_ext_set_pat);
            break;
        default:
            free(payload);
            return EINVAL;
        }
        ext = (const struct i915_user_extension *)(uintptr_t)ext->next_extension;
    }

    *out_payload = payload;
    *out_size = size;
    return 0;
}

static uint32_t
gem_blob_flags(void)
{
    uint32_t blob_flags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE |
                          VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
    if (g_vdrm->supports_cross_device)
        blob_flags |= VIRTGPU_BLOB_FLAG_USE_CROSS_DEVICE;
    return blob_flags;
}

static int
i915_virtio_gem_create(struct drm_i915_gem_create *create)
{
    struct vdrm_device *vdev = g_vdrm;
    struct i915_ccmd_gem_create_req req;
    memset(&req, 0, sizeof(req));
    req.hdr.cmd = I915_CCMD_GEM_CREATE;
    req.hdr.len = sizeof(req);
    req.size = create->size;
    req.blob_id = ++vdev->next_blob_id;

    uint32_t handle = vdrm_bo_create(vdev, create->size, gem_blob_flags(),
                                     req.blob_id, &req.hdr);
    if (!handle)
        return ENOMEM;

    create->handle = handle;
    return 0;
}

static int
i915_virtio_gem_create_ext(struct drm_i915_gem_create_ext *create)
{
    struct vdrm_device *vdev = g_vdrm;
    uint8_t *payload = NULL;
    uint32_t ext_size = 0;
    int ret;

    ret = gem_create_ext_payload(create, &payload, &ext_size);
    if (ret != 0)
        return ret;

    uint32_t req_len = (uint32_t)sizeof(struct i915_ccmd_gem_create_ext_req) +
                       ext_size;
    uint8_t *buf = (uint8_t *)malloc(req_len);
    if (!buf) {
        free(payload);
        return ENOMEM;
    }

    struct i915_ccmd_gem_create_ext_req *req =
        (struct i915_ccmd_gem_create_ext_req *)buf;
    memset(req, 0, sizeof(*req));
    req->hdr.cmd = I915_CCMD_GEM_CREATE_EXT;
    req->hdr.len = req_len;
    req->size = create->size;
    req->gem_flags = create->flags;
    req->ext_size = ext_size;
    req->blob_id = ++vdev->next_blob_id;
    if (ext_size > 0)
        memcpy(req->payload, payload, ext_size);

    uint32_t handle = vdrm_bo_create(vdev, create->size, gem_blob_flags(),
                                     req->blob_id, &req->hdr);

    free(payload);
    free(buf);
    if (!handle)
        return ENOMEM;

    create->handle = handle;
    return 0;
}

static int
i915_virtio_gem_close(struct drm_gem_close *close)
{
    vdrm_bo_close(g_vdrm, close->handle);
    return 0;
}

static int
i915_virtio_gem_context_create_ext(struct drm_i915_gem_context_create_ext *create)
{
    struct i915_virtio_ctx *ctx;
    const struct i915_user_extension *ext;
    uint32_t flags = create->flags;

    ctx = ctx_create(flags);
    if (!ctx)
        return ENOMEM;

    if (!(flags & I915_CONTEXT_CREATE_FLAGS_USE_EXTENSIONS)) {
        create->ctx_id = ctx->guest_id;
        return 0;
    }

    ext = (const struct i915_user_extension *)(uintptr_t)create->extensions;
    while (ext) {
        if (ext->name != I915_CONTEXT_CREATE_EXT_SETPARAM) {
            MOS_I915_VIRTIO_LOG("unsupported context create extension %u",
                                (unsigned)ext->name);
            ctx_destroy(ctx);
            return EINVAL;
        }

        const struct drm_i915_gem_context_create_ext_setparam *sp =
            (const struct drm_i915_gem_context_create_ext_setparam *)ext;
        if (!param_supported_at_create(sp->param.param)) {
            MOS_I915_VIRTIO_LOG("unsupported context create param %u",
                                (unsigned)sp->param.param);
            ctx_destroy(ctx);
            return EINVAL;
        }

        int ret = append_param(&ctx->params, &ctx->num_params,
                               sp->param.param, sp->param.size,
                               sp->param.value,
                               (const uint8_t *)(uintptr_t)sp->param.value);
        if (ret != 0) {
            ctx_destroy(ctx);
            return -ret;
        }

        ext = (const struct i915_user_extension *)(uintptr_t)ext->next_extension;
    }

    create->ctx_id = ctx->guest_id;
    return 0;
}

static int
i915_virtio_gem_context_destroy(struct drm_i915_gem_context_destroy *destroy)
{
    struct i915_virtio_ctx *ctx = ctx_lookup(destroy->ctx_id);
    if (!ctx)
        return EINVAL;

    ctx_destroy(ctx);
    return 0;
}

static int
i915_virtio_gem_context_param(unsigned long cmd,
                              struct drm_i915_gem_context_param *param)
{
    struct i915_virtio_ctx *ctx = ctx_lookup(param->ctx_id);
    if (!ctx)
        return EINVAL;

    if (IOC_MASKED(cmd) == IOC_MASKED(DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM)) {
        int ret = ctx_ensure_host(ctx);
        if (ret != 0)
            return -ret;

        struct i915_virtio_host_ctx *h = &g_host_ctxs[ctx->host_idx];
        uint32_t guest_id = param->ctx_id;
        param->ctx_id = h->host_ctx_id;

        ret = i915_virtio_simple_ioctl(cmd, param, false);
        param->ctx_id = guest_id;
        return ret;
    }

    /* SETPARAM */
    if (ctx->host_idx < 0) {
        /* Not yet created on the host: defer to creation time. */
        if (!param_supported_at_create(param->param)) {
            MOS_I915_VIRTIO_LOG("unsupported context param %u", (unsigned)param->param);
            return EINVAL;
        }

        if (param->param == I915_CONTEXT_PARAM_VM) {
            /* All contexts share the default VM; the parameter is not
             * needed on the host side.
             */
            return 0;
        }

        if (param->param == I915_CONTEXT_PARAM_ENGINES) {
            if (param->size == 0) {
                /* Revert to default engine map. */
                return 0;
            }

            uint32_t size = 0;
            uint32_t map_count = 0;
            struct i915_virtio_engine *map = NULL;
            uint8_t *payload = translate_engines_param(
                (const uint8_t *)(uintptr_t)param->value, param->size,
                &size, &map, &map_count);
            if (!payload) {
                MOS_I915_VIRTIO_LOG("failed to translate engines param");
                return EINVAL;
            }
            free(map);

            int ret = append_param(&ctx->params, &ctx->num_params,
                                   I915_CONTEXT_PARAM_ENGINES, size, 0,
                                   payload);
            free(payload);
            if (ret != 0)
                return -ret;
            return 0;
        }

        return append_param(&ctx->params, &ctx->num_params, param->param,
                            param->size, param->value,
                            (const uint8_t *)(uintptr_t)param->value);
    }

    if (!param_supported_post_create(param->param)) {
        MOS_I915_VIRTIO_LOG("unsupported context param %u", (unsigned)param->param);
        return EINVAL;
    }

    int ret = ctx_ensure_host(ctx);
    if (ret != 0)
        return -ret;

    struct i915_virtio_host_ctx *h = &g_host_ctxs[ctx->host_idx];
    uint32_t guest_id = param->ctx_id;
    param->ctx_id = h->host_ctx_id;

    ret = i915_virtio_simple_ioctl(cmd, param, false);
    param->ctx_id = guest_id;
    return ret;
}

static int
i915_virtio_gem_vm_control(unsigned long cmd, struct drm_i915_gem_vm_control *vm)
{
    if (vm->extensions) {
        MOS_I915_VIRTIO_LOG("unsupported vm extension");
        return EINVAL;
    }
    if (vm->flags) {
        MOS_I915_VIRTIO_LOG("unsupported vm flags");
        return EINVAL;
    }

    return i915_virtio_simple_ioctl(cmd, vm, false);
}

static int
i915_virtio_get_reset_stats(struct drm_i915_reset_stats *stats)
{
    struct vdrm_device *vdev = g_vdrm;
    struct i915_virtio_ctx *ctx = ctx_lookup(stats->ctx_id);
    if (!ctx)
        return EINVAL;

    int ret = ctx_ensure_host(ctx);
    if (ret != 0)
        return -ret;

    struct i915_virtio_host_ctx *h = &g_host_ctxs[ctx->host_idx];
    uint32_t host_id = h->host_ctx_id;

    struct i915_shmem *shmem = (struct i915_shmem *)vdev->shmem;
    if (host_id >= 64 ||
        !(*(volatile uint64_t *)&shmem->banned_ctx_mask & (1ULL << host_id)))
        return 0;

    uint32_t guest_id = stats->ctx_id;
    stats->ctx_id = host_id;
    ret = i915_virtio_simple_ioctl(DRM_IOCTL_I915_GET_RESET_STATS, stats,
                                   false);
    stats->ctx_id = guest_id;
    return ret;
}

static int
i915_virtio_gem_busy(struct drm_i915_gem_busy *busy)
{
    struct vdrm_device *vdev = g_vdrm;
    struct drm_virtgpu_3d_wait virt_wait;
    memset(&virt_wait, 0, sizeof(virt_wait));
    virt_wait.handle = busy->handle;
    virt_wait.flags = VIRTGPU_WAIT_NOWAIT;

    int ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_WAIT, &virt_wait);
    if (ret) {
        if (errno == EBUSY) {
            errno = 0;
            busy->busy = 1;
            return 0;
        }
        return errno ? errno : EIO;
    }

    busy->busy = 0;
    return 0;
}

static int64_t
time_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* DRM_IOCTL_VIRTGPU_WAIT has no timeout parameter, so finite timeouts
 * are implemented by polling.
 */
static int
i915_virtio_gem_wait(struct drm_i915_gem_wait *wait)
{
    struct vdrm_device *vdev = g_vdrm;
    struct drm_virtgpu_3d_wait virt_wait;
    int ret;

    memset(&virt_wait, 0, sizeof(virt_wait));
    virt_wait.handle = wait->bo_handle;

    if (!wait->timeout_ns)
        virt_wait.flags = VIRTGPU_WAIT_NOWAIT;
    else if (wait->timeout_ns > 0) {
        /* Finite timeout: poll until the deadline. */
        int64_t deadline = time_ns() + wait->timeout_ns;
        do {
            virt_wait.flags = VIRTGPU_WAIT_NOWAIT;
            ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_WAIT, &virt_wait);
            if (ret) {
                if (errno == EBUSY) {
                    if (time_ns() >= deadline)
                        return ETIME;
                    sched_yield();
                    continue;
                }
                return errno ? errno : EIO;
            }
            return 0;
        } while (true);
    }

    /* timeout_ns < 0: infinite wait */
    ret = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_WAIT, &virt_wait);
    if (ret) {
        if (errno == EBUSY)
            return ETIME;
        return errno ? errno : EIO;
    }

    return 0;
}

static int
i915_virtio_gem_set_mmap_mode(uint32_t res_id, uint32_t flags)
{
    struct vdrm_device *vdev = g_vdrm;
    struct i915_ccmd_gem_set_mmap_mode_req req;
    memset(&req, 0, sizeof(req));
    req.hdr.cmd = I915_CCMD_GEM_SET_MMAP_MODE;
    req.hdr.len = sizeof(req);
    req.res_id = res_id;
    req.flags = flags;

    int err = vdrm_send_req(vdev, &req.hdr, false);
    if (err)
        return errno ? errno : EIO;

    return 0;
}

static int
i915_virtio_gem_mmap_offset(struct drm_i915_gem_mmap_offset *mmap_offset)
{
    uint32_t res_id = vdrm_handle_to_res_id(g_vdrm, mmap_offset->handle);
    if (!res_id)
        return EINVAL;

    /* The actual mapping is done by mos_i915_gem_mmap(). */
    return i915_virtio_gem_set_mmap_mode(res_id, (uint32_t)mmap_offset->flags);
}

static int
i915_virtio_gem_mmap_gtt(struct drm_i915_gem_mmap_gtt *mmap_gtt)
{
    uint32_t res_id = vdrm_handle_to_res_id(g_vdrm, mmap_gtt->handle);
    if (!res_id)
        return EINVAL;

    /* The actual mapping is done by mos_i915_gem_mmap(). */
    return i915_virtio_gem_set_mmap_mode(res_id, I915_MMAP_OFFSET_GTT);
}

static int
i915_virtio_gem_mmap(struct drm_i915_gem_mmap *mmap)
{
    uint32_t res_id = vdrm_handle_to_res_id(g_vdrm, mmap->handle);
    if (!res_id)
        return EINVAL;

    int ret = i915_virtio_gem_set_mmap_mode(
        res_id, (mmap->flags & I915_MMAP_WC) ? I915_MMAP_OFFSET_WC
                                             : I915_MMAP_OFFSET_WB);
    if (ret != 0)
        return ret;

    void *map = mos_i915_gem_mmap(g_vdrm_fd, mmap->handle, mmap->size);
    if (map == MAP_FAILED)
        return ENOMEM;

    mmap->addr_ptr = (uint64_t)(uintptr_t)map;
    return 0;
}

static int
i915_virtio_gem_execbuffer2(struct drm_i915_gem_execbuffer2 *exec)
{
    struct vdrm_device *vdev = g_vdrm;
    struct i915_virtio_ctx *ctx;
    struct i915_virtio_host_ctx *h;
    struct i915_ccmd_gem_execbuffer2_rsp *rsp;
    struct drm_i915_gem_exec_object2 *buffers;
    uint32_t *bo_handles = NULL;
    uint8_t *buf = NULL;
    uint64_t flags = exec->flags;
    uint64_t host_flags;
    uint64_t ring_idx;
    size_t buffers_size, relocs_size;
    uint32_t relocs_count = 0;
    uint32_t buffer_count = exec->buffer_count;
    int in_fence_fd = -1;
    int out_fence_fd = -1;
    bool need_out_fence = !!(flags & I915_EXEC_FENCE_OUT);
    int ret = 0;
    int err;

    if (buffer_count == 0) {
        MOS_I915_VIRTIO_LOG("execbuffer2 with no objects");
        return EINVAL;
    }

    ctx = ctx_lookup((uint32_t)exec->rsvd1);
    if (!ctx) {
        MOS_I915_VIRTIO_LOG("execbuffer2 on unknown context %u",
                            (uint32_t)exec->rsvd1);
        return EINVAL;
    }

    err = ctx_ensure_host(ctx);
    if (err != 0)
        return -err;

    h = &g_host_ctxs[ctx->host_idx];

    /* Reject submissions on contexts banned by the host after a GPU
     * reset, mirroring the i915 kernel behavior.
     */
    {
        struct i915_shmem *shmem = (struct i915_shmem *)vdev->shmem;
        if (h->host_ctx_id < 64 &&
            (*(volatile uint64_t *)&shmem->banned_ctx_mask &
             (1ULL << h->host_ctx_id)))
            return EIO;
    }

    buffers = (struct drm_i915_gem_exec_object2 *)(uintptr_t)exec->buffers_ptr;

    for (uint32_t i = 0; i < buffer_count; i++)
        relocs_count += buffers[i].relocation_count;

    buffers_size = sizeof(*buffers) * buffer_count;
    relocs_size = sizeof(struct drm_i915_gem_relocation_entry) * relocs_count;

    size_t req_len = sizeof(struct i915_ccmd_gem_execbuffer2_req) +
                     buffers_size + relocs_size;

    buf = (uint8_t *)malloc(req_len);
    if (!buf)
        return ENOMEM;

    bo_handles = (uint32_t *)malloc(sizeof(uint32_t) * buffer_count);
    if (!bo_handles) {
        free(buf);
        return ENOMEM;
    }

    struct i915_ccmd_gem_execbuffer2_req *req =
        (struct i915_ccmd_gem_execbuffer2_req *)buf;
    memset(req, 0, sizeof(*req));
    memcpy(req->payload, buffers, buffers_size);

    struct drm_i915_gem_exec_object2 *out_buffers =
        (struct drm_i915_gem_exec_object2 *)req->payload;
    struct drm_i915_gem_relocation_entry *out_relocs =
        (struct drm_i915_gem_relocation_entry *)(req->payload + buffers_size);

    /* Pre-compute res_ids for all buffers so we can convert both the
     * buffer array handles and the relocation target handles.
     */
    uint32_t res_ids[64];
    for (uint32_t i = 0; i < buffer_count && i < ARRAY_SIZE(res_ids); i++) {
        res_ids[i] = vdrm_handle_to_res_id(vdev, buffers[i].handle);
    }

    for (uint32_t i = 0; i < buffer_count; i++) {
        if (buffers[i].relocation_count) {
            if (buffers[i].relocs_ptr == 0) {
                free(bo_handles);
                free(buf);
                return EINVAL;
            }
            memcpy(out_relocs,
                   (struct drm_i915_gem_relocation_entry *)(uintptr_t)buffers[i].relocs_ptr,
                   sizeof(*out_relocs) * buffers[i].relocation_count);

            /* Translate guest GEM target handles to host resource IDs
             * so the host can look them up in its resource table.
             */
            for (uint32_t j = 0; j < buffers[i].relocation_count; j++) {
                for (uint32_t k = 0; k < buffer_count; k++) {
                    if (out_relocs[j].target_handle == buffers[k].handle) {
                        out_relocs[j].target_handle = res_ids[k];
                        break;
                    }
                }
            }
            out_relocs += buffers[i].relocation_count;
        }

        bo_handles[i] = buffers[i].handle;
        out_buffers[i].handle = res_ids[i];
    }

    host_flags = translate_exec_flags(h, flags);

    /* Fence in/out is emulated by the virtgpu execbuffer ioctl. */
    host_flags &= ~((uint64_t)I915_EXEC_FENCE_IN |
                    (uint64_t)I915_EXEC_FENCE_OUT |
                    (uint64_t)I915_EXEC_FENCE_SUBMIT);

    req->hdr.cmd = (vdev->caps.wire_format_version == 1)
                       ? I915_CCMD_GEM_EXECBUFFER2
                       : I915_CCMD_GEM_EXECBUFFER2_V2;
    req->hdr.len = (uint32_t)req_len;
    req->flags = host_flags;
    req->context_id = h->host_ctx_id;
    req->buffer_count = buffer_count;
    req->batch_start_offset = exec->batch_start_offset;
    req->batch_len = exec->batch_len;
    req->relocs_count = relocs_count;

    rsp = (struct i915_ccmd_gem_execbuffer2_rsp *)
        vdrm_alloc_rsp(vdev, &req->hdr, sizeof(*rsp));

    /* 0 reserved for cpu sync */
    ring_idx = 1 + (host_flags & I915_EXEC_RING_MASK);
    if (vdev->caps.wire_format_version > 1)
        ring_idx += ((uint64_t)h->host_ctx_id - 1) * 8;

    if (flags & (I915_EXEC_FENCE_IN | I915_EXEC_FENCE_SUBMIT))
        in_fence_fd = (int)(exec->rsvd2 & 0xffffffffu);

    {
        struct drm_virtgpu_execbuffer eb;
        memset(&eb, 0, sizeof(eb));
        eb.flags = VIRTGPU_EXECBUF_RING_IDX;
        if (in_fence_fd >= 0)
            eb.flags |= VIRTGPU_EXECBUF_FENCE_FD_IN;
        if (need_out_fence)
            eb.flags |= VIRTGPU_EXECBUF_FENCE_FD_OUT;
        eb.size = (uint32_t)req_len;
        eb.command = (uintptr_t)req;
        eb.bo_handles = (uintptr_t)bo_handles;
        eb.num_bo_handles = buffer_count;
        eb.fence_fd = in_fence_fd;
        eb.ring_idx = (uint32_t)ring_idx;

        pthread_mutex_lock(&vdev->eb_lock);
        vdrm_flush_locked(vdev, NULL);
        req->hdr.seqno = ++vdev->next_seqno;

        err = drmIoctl(vdev->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &eb);
        pthread_mutex_unlock(&vdev->eb_lock);

        if (!err && need_out_fence)
            out_fence_fd = eb.fence_fd;
    }

    /* Wait until the host has processed the ccmd and written the
     * response.
     */
    vdrm_host_sync(vdev, &req->hdr);

    if (err) {
        errno = errno ? errno : EIO;
        ret = -1;
        goto out;
    }

    if (rsp->ret) {
        if (out_fence_fd >= 0)
            close(out_fence_fd);
        errno = rsp->ret;
        ret = -1;
        goto out;
    }

    if (need_out_fence && out_fence_fd >= 0)
        exec->rsvd2 = (uint64_t)(uint32_t)out_fence_fd << 32;

out:
    free(bo_handles);
    free(buf);
    return ret;
}

/* ======================================================================
 * ioctl dispatch
 * ====================================================================== */

int
mos_i915_ioctl(int fd, unsigned long request, void *arg)
{
    struct vdrm_device *vdev = NULL;
    int orig_errno = errno;
    int ret = 0;

    /* Lazy init: if the fd matches a known virtio-gpu i915 fd but vdrm
     * isn't connected yet, connect now.  The probe result is memoized
     * for fds that are not virtio-gpu i915 devices to avoid a syscall
     * per ioctl on bare metal.
     */
    if (!g_vdrm && g_vdrm_fd < 0 && g_non_virtio_fd != fd) {
        if (mos_i915_virtio_probe(fd)) {
            if (!g_vdrm && g_vdrm_fd < 0 && mos_i915_virtio_probe(fd))
                mos_i915_virtio_init(fd);
        } else {
            g_non_virtio_fd = fd;
        }
    }

    /* Fast path: if not on virtio, just call the real ioctl */
    pthread_mutex_lock(&g_vdrm_global_lock);
    if (!g_vdrm || g_vdrm_fd != fd) {
        pthread_mutex_unlock(&g_vdrm_global_lock);
        errno = orig_errno;
        return drmIoctl(fd, request, arg);
    }
    vdev = g_vdrm;
    pthread_mutex_unlock(&g_vdrm_global_lock);

    switch (IOC_MASKED(request)) {
    case IOC_MASKED(DRM_IOCTL_I915_GEM_CREATE):
        ret = i915_virtio_gem_create((struct drm_i915_gem_create *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_CREATE_EXT):
        ret = i915_virtio_gem_create_ext((struct drm_i915_gem_create_ext *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_GEM_CLOSE):
        ret = i915_virtio_gem_close((struct drm_gem_close *)arg);
        break;

    /* CONTEXT_CREATE and CONTEXT_CREATE_EXT share the same ioctl
     * number; distinguish them by argument size.
     */
    case IOC_MASKED(DRM_IOCTL_I915_GEM_CONTEXT_CREATE):
        if (_IOC_SIZE(request) <= sizeof(struct drm_i915_gem_context_create)) {
            /* Plain context create: no flags, no extensions. */
            struct drm_i915_gem_context_create_ext create;
            memset(&create, 0, sizeof(create));
            ret = i915_virtio_gem_context_create_ext(&create);
            if (ret == 0)
                ((struct drm_i915_gem_context_create *)arg)->ctx_id =
                    create.ctx_id;
        } else {
            ret = i915_virtio_gem_context_create_ext(
                (struct drm_i915_gem_context_create_ext *)arg);
        }
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_CONTEXT_DESTROY):
        ret = i915_virtio_gem_context_destroy(
            (struct drm_i915_gem_context_destroy *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM):
    case IOC_MASKED(DRM_IOCTL_I915_GEM_CONTEXT_SETPARAM):
        ret = i915_virtio_gem_context_param(
            request, (struct drm_i915_gem_context_param *)arg);
        break;

    /* EXECBUFFER2 and EXECBUFFER2_WR share the ioctl number but differ
     * in direction bits.
     */
    case IOC_MASKED(DRM_IOCTL_I915_GEM_EXECBUFFER2):
    case IOC_MASKED(DRM_IOCTL_I915_GEM_EXECBUFFER2_WR):
        ret = i915_virtio_gem_execbuffer2(
            (struct drm_i915_gem_execbuffer2 *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GETPARAM):
        ret = i915_virtio_getparam((struct drm_i915_getparam *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_QUERY):
        ret = i915_virtio_queryparam((struct drm_i915_query *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_MMAP):
        ret = i915_virtio_gem_mmap((struct drm_i915_gem_mmap *)arg);
        break;

    /* MMAP_OFFSET and MMAP_GTT share the same ioctl number; distinguish
     * them by argument size.
     */
    case IOC_MASKED(DRM_IOCTL_I915_GEM_MMAP_GTT):
        if (_IOC_SIZE(request) <= sizeof(struct drm_i915_gem_mmap_gtt))
            ret = i915_virtio_gem_mmap_gtt((struct drm_i915_gem_mmap_gtt *)arg);
        else
            ret = i915_virtio_gem_mmap_offset(
                (struct drm_i915_gem_mmap_offset *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_BUSY):
        ret = i915_virtio_gem_busy((struct drm_i915_gem_busy *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_WAIT):
        ret = i915_virtio_gem_wait((struct drm_i915_gem_wait *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_MADVISE):
        /* Nothing to do; the host keeps objects around. */
        ret = 0;
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_SET_DOMAIN):
        ret = i915_virtio_simple_ioctl(request, arg, true);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_SET_TILING):
    case IOC_MASKED(DRM_IOCTL_I915_GEM_GET_TILING):
        ret = i915_virtio_simple_ioctl(request, arg, true);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_SW_FINISH): {
        struct drm_gem_close *close_arg = (struct drm_gem_close *)arg;
        struct drm_i915_gem_set_domain set_domain;
        memset(&set_domain, 0, sizeof(set_domain));
        set_domain.handle = close_arg->handle;
        set_domain.read_domains = I915_GEM_DOMAIN_GTT;
        set_domain.write_domain = I915_GEM_DOMAIN_GTT;
        ret = i915_virtio_simple_ioctl(DRM_IOCTL_I915_GEM_SET_DOMAIN,
                                       &set_domain, true);
        break;
    }

    case IOC_MASKED(DRM_IOCTL_I915_GEM_VM_CREATE):
    case IOC_MASKED(DRM_IOCTL_I915_GEM_VM_DESTROY):
        ret = i915_virtio_gem_vm_control(
            request, (struct drm_i915_gem_vm_control *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GET_RESET_STATS):
        ret = i915_virtio_get_reset_stats(
            (struct drm_i915_reset_stats *)arg);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_REG_READ):
    case IOC_MASKED(DRM_IOCTL_I915_GEM_GET_APERTURE):
        ret = i915_virtio_simple_ioctl(request, arg, false);
        break;

    case IOC_MASKED(DRM_IOCTL_I915_GEM_USERPTR):
        errno = ENODEV;
        ret = -1;
        break;

    case IOC_MASKED(DRM_IOCTL_GEM_FLINK):
    case IOC_MASKED(DRM_IOCTL_GEM_OPEN):
        errno = EINVAL;
        ret = -1;
        break;

    default:
        /* SYNCOBJ_*, PRIME_*, etc. -- pass through to kernel.
         * Use vdev->fd since handles/syncobjs live on our context fd
         * (which may be a reopened fd if EEXIST was encountered).
         */
        ret = drmIoctl(vdev->fd, request, arg);
        break;
    }

    if (ret != 0) {
        if (ret > 0)
            errno = ret;
        return -1;
    }

    errno = orig_errno;
    return 0;
}

/* ======================================================================
 * Public API
 * ====================================================================== */

bool
mos_i915_virtio_probe(int fd)
{
    struct virgl_renderer_capset_drm caps;

    if (vdrm_get_capset(fd, &caps) != 0)
        return false;

    return caps.context_type == VIRTGPU_DRM_CONTEXT_I915;
}

int
mos_i915_virtio_init(int fd)
{
    pthread_mutex_lock(&g_vdrm_global_lock);

    if (g_vdrm && g_vdrm_fd == fd) {
        /* Already initialized for this fd */
        pthread_mutex_unlock(&g_vdrm_global_lock);
        return 0;
    }

    if (g_vdrm) {
        /* Different fd -- close old one first */
        reset_context_state();
        vdrm_disconnect(g_vdrm);
        g_vdrm = NULL;
        g_vdrm_fd = -1;
    }

    g_non_virtio_fd = -1;

    struct vdrm_device *vdev = vdrm_connect(fd);
    if (!vdev) {
        pthread_mutex_unlock(&g_vdrm_global_lock);
        return -ENODEV;
    }

    g_vdrm = vdev;
    g_vdrm_fd = fd;

    /* Cross-device feature is optional.  It enables sharing dma-bufs
     * with other virtio devices, like virtio-wl or virtio-video used
     * by ChromeOS VMs.  Qemu doesn't support cross-device sharing.
     */
    {
        struct drm_virtgpu_getparam gp;
        memset(&gp, 0, sizeof(gp));
        uint64_t val = 0;
        gp.param = VIRTGPU_PARAM_CROSS_DEVICE;
        gp.value = (uintptr_t)&val;
        if (drmIoctl(fd, DRM_IOCTL_VIRTGPU_GETPARAM, &gp) == 0 && val)
            vdev->supports_cross_device = true;
    }

    pthread_mutex_unlock(&g_vdrm_global_lock);
    return 0;
}

void
mos_i915_virtio_fini(int fd)
{
    pthread_mutex_lock(&g_vdrm_global_lock);

    if (g_vdrm && g_vdrm_fd == fd) {
        reset_context_state();
        vdrm_disconnect(g_vdrm);
        g_vdrm = NULL;
        g_vdrm_fd = -1;
    }

    if (g_non_virtio_fd == fd)
        g_non_virtio_fd = -1;

    pthread_mutex_unlock(&g_vdrm_global_lock);
}

bool
mos_i915_virtio_is_active(int fd)
{
    return g_vdrm != NULL && g_vdrm_fd == fd;
}

void *
mos_i915_gem_mmap(int fd, uint32_t handle, size_t size)
{
    if (!g_vdrm || g_vdrm_fd != fd)
        return MAP_FAILED;

    struct drm_virtgpu_map req;
    memset(&req, 0, sizeof(req));
    req.handle = handle;

    int vfd = g_vdrm->fd;
    int ret = drmIoctl(vfd, DRM_IOCTL_VIRTGPU_MAP, &req);
    if (ret) {
        MOS_I915_VIRTIO_LOG("mos_i915_gem_mmap: VIRTGPU_MAP failed: %s",
                            strerror(errno));
        return MAP_FAILED;
    }

    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, vfd, req.offset);
    if (map == MAP_FAILED) {
        MOS_I915_VIRTIO_LOG("mos_i915_gem_mmap: mmap failed: %s",
                            strerror(errno));
    }
    return map;
}
