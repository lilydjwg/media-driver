/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Virtio-GPU native context support for i915 in the iHD media driver.
 *
 * When running inside a VM with virtio-gpu and the host exposes an
 * Intel GPU via the i915 native context (VIRTGPU_DRM_CONTEXT_I915
 * capset), this layer transparently routes i915 DRM ioctls through the
 * vdrm protocol so that mos_bufmgr.c works without modification of its
 * call flow.
 *
 * The wire protocol and host side semantics are implemented by
 * virglrenderer's src/drm/i915/i915_ccmd.c; this client side mirrors
 * mesa's src/intel/dev/virtio/i915_virtio_ccmd.c.
 */

#ifndef MOS_I915_VIRTIO_H
#define MOS_I915_VIRTIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Probe whether @p fd is a virtio-gpu device whose host exposes an
 * i915 native context (VIRTGPU_DRM_CONTEXT_I915 capset).
 *
 * Returns true if the fd is suitable for virtio i915 passthrough.
 */
bool mos_i915_virtio_probe(int fd);

/**
 * Initialize the vdrm proxy connection on @p fd.
 * Call once during bufmgr init after mos_i915_virtio_probe() returned true.
 *
 * @return 0 on success, negative errno on failure.
 */
int mos_i915_virtio_init(int fd);

/**
 * Clean up the vdrm proxy for @p fd.
 */
void mos_i915_virtio_fini(int fd);

/**
 * Check whether the vdrm proxy is active on @p fd.
 */
bool mos_i915_virtio_is_active(int fd);

/**
 * ioctl wrapper -- routes i915 DRM ioctls through the vdrm protocol
 * when running on virtio-gpu.  Falls through to drmIoctl() for
 * non-i915 ioctls and when not on virtio-gpu.
 */
int mos_i915_ioctl(int fd, unsigned long request, void *arg);

/**
 * mmap wrapper for GEM objects -- uses VIRTGPU_MAP on virtio-gpu,
 * returns MAP_FAILED on error.
 */
void *mos_i915_gem_mmap(int fd, uint32_t handle, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* MOS_I915_VIRTIO_H */
