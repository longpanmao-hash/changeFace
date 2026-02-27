import argparse
import os
import sys

import cv2
import insightface
import numpy as np
from insightface.model_zoo import get_model
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None
try:
    import mediapipe as mp
except Exception:
    mp = None


def ensure_png(path: str, label: str) -> None:
    if not path.lower().endswith(".png"):
        raise ValueError(f"{label} must be a .png file: {path}")


def pick_largest_face(faces):
    if not faces:
        return None
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]


def average_source_embedding(app, source_paths):
    vectors = []
    for p in source_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        faces = app.get(img)
        face = pick_largest_face(faces)
        if face is None:
            continue
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb) + 1e-8
        vectors.append(emb / n)
    if not vectors:
        return None
    mean_vec = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    return mean_vec


def _face_points(face):
    pts = getattr(face, "landmark_2d_106", None)
    if pts is not None and len(pts) >= 20:
        return np.asarray(pts, dtype=np.float32)
    return np.asarray(face.kps, dtype=np.float32)


def _bbox_mask(shape, bbox, expand=0.18):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * (1.0 + expand)
    bh = (y2 - y1) * (1.0 + expand)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(cx), int(cy))
    axes = (max(1, int(bw / 2)), max(1, int(bh / 2)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _head_mask(shape, bbox):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    # include more forehead/hair and slightly less neck
    center = (int(cx), int(cy - 0.06 * bh))
    axes = (max(1, int(0.62 * bw)), max(1, int(0.86 * bh)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # trim lower neck area to reduce shirt bleeding
    y_cut = int(cy + 0.44 * bh)
    if y_cut < h:
      mask[y_cut:, :] = 0
    return mask


def _head_mask_from_landmarks(shape, bbox, pts):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_i = np.asarray(pts, dtype=np.int32)
    if len(pts_i) >= 6:
        hull = cv2.convexHull(pts_i)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        mask = _bbox_mask(shape, bbox, expand=0.1)

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    # add controlled top-hair region without bringing too much background
    hair = np.zeros((h, w), dtype=np.uint8)
    center = (int(cx), int(y1 + 0.22 * bh))
    axes = (max(1, int(0.52 * bw)), max(1, int(0.42 * bh)))
    cv2.ellipse(hair, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.bitwise_or(mask, hair)

    k = max(5, int(max(bw, bh) * 0.035))
    kernel = np.ones((k, k), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    y_cut = int((y1 + y2) / 2.0 + 0.62 * bh)
    if y_cut < h:
        mask[y_cut:, :] = 0
    return mask


def _person_mask(source_bgr):
    # 1) try MediaPipe selfie segmentation first (cleaner person foreground)
    if mp is not None:
        try:
            rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
                res = seg.process(rgb)
            if res.segmentation_mask is not None:
                m = (res.segmentation_mask * 255.0).astype(np.uint8)
                return m
        except Exception:
            pass

    # 2) fallback to rembg
    if rembg_remove is not None:
        try:
            rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            m = rembg_remove(rgb, only_mask=True)
            if m is None:
                return None
            if len(m.shape) == 3:
                m = m[:, :, 0]
            m = np.asarray(m, dtype=np.uint8)
            if np.max(m) <= 1:
                m = (m * 255).astype(np.uint8)
            return m
        except Exception:
            return None
    return None


def _largest_component(mask, seed_xy=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    if seed_xy is not None:
        sx, sy = int(seed_xy[0]), int(seed_xy[1])
        if 0 <= sy < labels.shape[0] and 0 <= sx < labels.shape[1]:
            label = labels[sy, sx]
            if label > 0:
                out = np.zeros_like(mask, dtype=np.uint8)
                out[labels == label] = 255
                return out
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == best] = 255
    return out


def _source_head_cutout_mask(source_bgr, src_face):
    h, w = source_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in src_face.bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # broader ROI to keep full hairstyle around the face
    rx1 = max(0, int(cx - 1.20 * bw))
    rx2 = min(w, int(cx + 1.20 * bw))
    ry1 = max(0, int(y1 - 1.15 * bh))
    ry2 = min(h, int(cy + 1.05 * bh))
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[ry1:ry2, rx1:rx2] = 255

    envelope = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        envelope,
        (int(cx), int(cy - 0.03 * bh)),
        (max(1, int(1.02 * bw)), max(1, int(1.20 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    base = cv2.bitwise_and(envelope, roi)

    person_mask = _person_mask(source_bgr)
    if person_mask is not None:
        fg = np.where(person_mask > 110, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        fg = _largest_component(fg, seed_xy=(cx, cy))
        bh2 = max(2, int(h * 0.08))
        bw2 = max(2, int(w * 0.08))
        border_samples = np.concatenate(
            [
                source_bgr[:bh2, :, :].reshape(-1, 3),
                source_bgr[:, :bw2, :].reshape(-1, 3),
                source_bgr[:, -bw2:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = np.median(border_samples, axis=0).astype(np.float32)
        diff = np.linalg.norm(source_bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
        fg_color = np.where(diff > 20.0, 255, 0).astype(np.uint8)
        fg = cv2.bitwise_and(fg, fg_color)
        base = cv2.bitwise_and(fg, base)
    else:
        # fallback background suppression by source border color
        bh = max(2, int(h * 0.08))
        bw2 = max(2, int(w * 0.08))
        border_samples = np.concatenate(
            [
                source_bgr[:bh, :, :].reshape(-1, 3),
                source_bgr[:, :bw2, :].reshape(-1, 3),
                source_bgr[:, -bw2:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = np.median(border_samples, axis=0).astype(np.float32)
        diff = np.linalg.norm(source_bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
        fg = np.where(diff > 28.0, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        base = cv2.bitwise_and(base, fg)

    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    neck_remove = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        neck_remove,
        (int(cx), int(y2 + 0.16 * bh)),
        (max(1, int(0.44 * bw)), max(1, int(0.28 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    base = cv2.subtract(base, neck_remove)
    base = _largest_component(base, seed_xy=(cx, cy))
    return base


def _color_match_lab(src_bgr, ref_bgr, mask):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 10
    if m.sum() < 50:
        return src_bgr
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:, :, c][m]
        r = ref_lab[:, :, c][m]
        s_mean = float(np.mean(s))
        r_mean = float(np.mean(r))
        s_std = float(np.std(s)) + 1e-6
        r_std = float(np.std(r)) + 1e-6
        out[:, :, c] = (src_lab[:, :, c] - s_mean) * (r_std / s_std) + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def fullface_blend(source_bgr, base_bgr, src_face, tgt_face):
    h, w = base_bgr.shape[:2]
    src_pts = _face_points(src_face)
    dst_pts = _face_points(tgt_face)
    if src_pts.shape[0] != dst_pts.shape[0]:
        n = min(src_pts.shape[0], dst_pts.shape[0])
        src_pts = src_pts[:n]
        dst_pts = dst_pts[:n]
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return base_bgr

    warped_src = cv2.warpAffine(source_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    dst_pts_i = dst_pts.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(dst_pts_i) >= 10:
        hull = cv2.convexHull(dst_pts_i)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        mask = _bbox_mask(base_bgr.shape, tgt_face.bbox, expand=0.22)

    # expand jaw/cheek coverage
    k = max(7, int(max(tgt_face.bbox[2] - tgt_face.bbox[0], tgt_face.bbox[3] - tgt_face.bbox[1]) * 0.06))
    kernel = np.ones((k, k), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # remove central facial area to avoid double eyes/nose/mouth ghosting
    inner = _bbox_mask(
        base_bgr.shape,
        tgt_face.bbox,
        expand=-0.35 if (tgt_face.bbox[2] - tgt_face.bbox[0]) > 0 else 0,
    )
    mask = cv2.subtract(mask, inner)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, k // 3), sigmaY=max(3, k // 3))
    warped_src = _color_match_lab(warped_src, base_bgr, mask)

    alpha = (mask.astype(np.float32) / 255.0) ** 1.6
    alpha = (alpha * 0.45)[:, :, None]
    mixed = warped_src.astype(np.float32) * alpha + base_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def headpaste_blend(source_bgr, target_bgr, src_face, tgt_face, head_strength=0.65):
    h, w = target_bgr.shape[:2]
    # use 5-point similarity transform for stability and to avoid warped ghosting
    src_pts = np.asarray(src_face.kps, dtype=np.float32)
    dst_pts = np.asarray(tgt_face.kps, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return target_bgr

    warped_src = cv2.warpAffine(
        source_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    src_head_mask = _head_mask(source_bgr.shape, src_face.bbox)
    person_mask = _person_mask(source_bgr)
    if person_mask is not None:
        # keep only confident person foreground to remove source background halos
        person_bin = np.where(person_mask > 140, 255, 0).astype(np.uint8)
        person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        src_head_mask = cv2.bitwise_and(src_head_mask, person_bin)
    warped_mask = cv2.warpAffine(src_head_mask, M, (w, h), flags=cv2.INTER_LINEAR)

    # remove source background-like pixels from warped mask to avoid top halo ghosting
    sh, sw = source_bgr.shape[:2]
    bw = max(2, int(sw * 0.08))
    bh = max(2, int(sh * 0.08))
    border_samples = np.concatenate(
        [
            source_bgr[:bh, :, :].reshape(-1, 3),
            source_bgr[: (2 * bh), :bw, :].reshape(-1, 3),
            source_bgr[: (2 * bh), -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(border_samples, axis=0).astype(np.float32)
    diff = np.linalg.norm(warped_src.astype(np.float32) - bg_color[None, None, :], axis=2)
    bg_like = diff < 36.0
    warped_mask[bg_like] = 0
    warped_mask = cv2.morphologyEx(
        warped_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=2
    )
    # remove central face region to prevent facial ghosting
    x1, y1, x2, y2 = tgt_face.bbox
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    inner_face = np.zeros_like(warped_mask, dtype=np.uint8)
    cv2.ellipse(
        inner_face,
        (cx, int(cy + 0.03 * bh)),
        (max(1, int(0.43 * bw)), max(1, int(0.52 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    warped_mask = cv2.subtract(warped_mask, inner_face)

    # tighten mask to avoid bringing source background
    face_w = max(1.0, float(tgt_face.bbox[2] - tgt_face.bbox[0]))
    face_h = max(1.0, float(tgt_face.bbox[3] - tgt_face.bbox[1]))
    k = max(5, int(max(face_w, face_h) * 0.03))
    erode_kernel = np.ones((k, k), dtype=np.uint8)
    soft_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=max(3, k), sigmaY=max(3, k))

    warped_src = _color_match_lab(warped_src, target_bgr, soft_mask)
    alpha = (soft_mask.astype(np.float32) / 255.0) ** 1.6
    alpha = np.clip(alpha * (0.45 + 0.45 * float(head_strength)), 0.0, 1.0)
    # suppress top halo by attenuating very upper head blend
    y0 = float(tgt_face.bbox[1])
    y1 = float(tgt_face.bbox[3])
    hh = max(1.0, y1 - y0)
    ys = np.arange(h, dtype=np.float32)
    t = (ys - (y0 + 0.05 * hh)) / (0.32 * hh)
    t = np.clip(t, 0.0, 1.0)
    top_fade = t * t * (3.0 - 2.0 * t)
    alpha = alpha * top_fade[:, None]
    # suppress chin/neck carry-over for smoother neck transition
    t_neck = (ys - (y0 + 0.66 * hh)) / (0.20 * hh)
    neck_fade = 1.0 - np.clip(t_neck, 0.0, 1.0)
    alpha = alpha * neck_fade[:, None]
    alpha = alpha[:, :, None]
    mixed = warped_src.astype(np.float32) * alpha + target_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def headreplace_blend(source_bgr, target_bgr, src_face, tgt_face):
    h, w = target_bgr.shape[:2]
    src_pts = np.asarray(src_face.kps, dtype=np.float32)
    dst_pts = np.asarray(tgt_face.kps, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return target_bgr

    warped_src = cv2.warpAffine(
        source_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    src_head_mask = _source_head_cutout_mask(source_bgr, src_face)
    warped_mask = cv2.warpAffine(
        src_head_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    x1, y1, x2, y2 = [float(v) for v in tgt_face.bbox]
    tbw = max(1.0, x2 - x1)
    tbh = max(1.0, y2 - y1)
    tcx = int((x1 + x2) / 2.0)
    tcy = int((y1 + y2) / 2.0)

    # recover hairstyle coverage from real warped source pixels only
    non_black = np.where(np.max(warped_src, axis=2) > 12, 255, 0).astype(np.uint8)
    valid_src = None
    src_person = _person_mask(source_bgr)
    if src_person is not None:
        src_person = np.where(src_person > 120, 255, 0).astype(np.uint8)
        src_person = cv2.morphologyEx(src_person, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        warped_person = cv2.warpAffine(
            src_person, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        valid_src = np.where(warped_person > 80, 255, 0).astype(np.uint8)
    else:
        sh, sw = source_bgr.shape[:2]
        sbh = max(2, int(sh * 0.08))
        sbw = max(2, int(sw * 0.08))
        border_samples = np.concatenate(
            [
                source_bgr[:sbh, :, :].reshape(-1, 3),
                source_bgr[:, :sbw, :].reshape(-1, 3),
                source_bgr[:, -sbw:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = np.median(border_samples, axis=0).astype(np.float32)
        diff = np.linalg.norm(warped_src.astype(np.float32) - bg_color[None, None, :], axis=2)
        valid_src = np.where(diff > 28.0, 255, 0).astype(np.uint8)
    valid_src = cv2.bitwise_and(valid_src, non_black)
    valid_src = cv2.morphologyEx(valid_src, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    valid_src = cv2.morphologyEx(valid_src, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    hair_zone = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        hair_zone,
        (tcx, int(tcy - 0.06 * tbh)),
        (max(1, int(0.82 * tbw)), max(1, int(0.96 * tbh))),
        0,
        0,
        360,
        255,
        -1,
    )
    side_zone = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        side_zone,
        (tcx, int(tcy + 0.10 * tbh)),
        (max(1, int(0.92 * tbw)), max(1, int(0.86 * tbh))),
        0,
        0,
        360,
        255,
        -1,
    )
    hair_zone = cv2.bitwise_or(hair_zone, side_zone)
    y_hair_limit = int(y2 + 0.12 * tbh)
    if y_hair_limit < h:
        hair_zone[y_hair_limit:, :] = 0
    hair_zone = cv2.bitwise_and(hair_zone, valid_src)
    warped_mask = cv2.bitwise_or(warped_mask, hair_zone)

    # suppress lower-neck transfer with curved mask (avoids horizontal seam lines)
    neck_guard = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        neck_guard,
        (tcx, int(y2 + 0.08 * tbh)),
        (max(1, int(0.48 * tbw)), max(1, int(0.30 * tbh))),
        0,
        0,
        360,
        255,
        -1,
    )
    warped_mask = cv2.subtract(warped_mask, neck_guard)
    rx1 = max(0, int(tcx - 0.56 * tbw))
    rx2 = min(w, int(tcx + 0.56 * tbw))
    ry1 = max(0, int(y2 - 0.02 * tbh))
    ry2 = min(h, int(y2 + 0.14 * tbh))
    if rx2 > rx1 and ry2 > ry1:
        seam_band = np.zeros((h, w), dtype=np.uint8)
        seam_band[ry1:ry2, rx1:rx2] = 255
        warped_mask = cv2.subtract(warped_mask, seam_band)

    grow_k = max(3, int(max(tbw, tbh) * 0.03))
    warped_mask = cv2.dilate(warped_mask, np.ones((grow_k, grow_k), np.uint8), iterations=1)
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    warped_mask = cv2.medianBlur(warped_mask, 5)
    warped_mask = np.where(warped_mask > 60, 255, 0).astype(np.uint8)
    warped_mask = _largest_component(
        warped_mask,
        seed_xy=((tgt_face.bbox[0] + tgt_face.bbox[2]) / 2.0, (tgt_face.bbox[1] + tgt_face.bbox[3]) / 2.0),
    )

    edge_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=max(2, grow_k // 2), sigmaY=max(2, grow_k // 2))

    warped_src = _color_match_lab(warped_src, target_bgr, edge_mask)

    # narrow feather band: interior is hard replacement, only boundary is blended
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5)
    feather_px = max(2.0, float(max(tbw, tbh)) * 0.010)
    alpha = np.clip(dist / feather_px, 0.0, 1.0)
    alpha[warped_mask == 0] = 0.0
    alpha = alpha[:, :, None]

    mixed = warped_src.astype(np.float32) * alpha + target_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def features_blend(swapped_bgr, target_bgr, tgt_face, strength=0.70):
    h, w = target_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in tgt_face.bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)

    mask = np.zeros((h, w), dtype=np.uint8)
    # central facial area (keep jawline/forehead mostly from target)
    cv2.ellipse(
        mask,
        (cx, int(cy + 0.06 * bh)),
        (max(1, int(0.40 * bw)), max(1, int(0.48 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    # emphasize eye/brow region
    cv2.ellipse(
        mask,
        (cx, int(cy - 0.16 * bh)),
        (max(1, int(0.43 * bw)), max(1, int(0.20 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    # emphasize mouth region
    cv2.ellipse(
        mask,
        (cx, int(cy + 0.23 * bh)),
        (max(1, int(0.30 * bw)), max(1, int(0.16 * bh))),
        0,
        0,
        360,
        255,
        -1,
    )
    # avoid neck leakage
    y_cut = int(y2 + 0.02 * bh)
    if y_cut < h:
        mask[y_cut:, :] = 0

    k = max(5, int(max(bw, bh) * 0.025))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, k), sigmaY=max(2, k))
    swapped_adj = _color_match_lab(swapped_bgr, target_bgr, mask)

    alpha = (mask.astype(np.float32) / 255.0) ** 1.5
    alpha = np.clip(alpha * (0.35 + 0.50 * float(strength)), 0.0, 0.88)
    alpha = alpha[:, :, None]
    mixed = swapped_adj.astype(np.float32) * alpha + target_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep face swap with InsightFace InSwapper")
    parser.add_argument("--source", required=True, help="source face image (.png)")
    parser.add_argument("--target", required=True, help="target image (.png)")
    parser.add_argument("--output", required=True, help="output image (.png)")
    parser.add_argument(
        "--swapper-model",
        default=os.path.join("deep", "models", "inswapper_128.onnx"),
        help="path to inswapper_128.onnx",
    )
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "fullface", "headpaste", "headreplace", "features"],
        help="standard: identity swap, fullface: stronger swap including face shape region, headpaste: blend source head/hair onto target, headreplace: directly replace target head with source head, features: swap facial traits while preserving target head shape/hair",
    )
    parser.add_argument(
        "--source-refs",
        default="",
        help="optional extra source PNGs split by comma, e.g. a.png,b.png,c.png",
    )
    parser.add_argument(
        "--head-strength",
        default="0.65",
        help="headpaste blend strength 0~1 (lower is more natural)",
    )
    args = parser.parse_args()

    ensure_png(args.source, "source")
    ensure_png(args.target, "target")
    ensure_png(args.output, "output")
    head_strength = float(args.head_strength)
    head_strength = min(1.0, max(0.0, head_strength))

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"source not found: {args.source}")
    if not os.path.exists(args.target):
        raise FileNotFoundError(f"target not found: {args.target}")
    if not os.path.exists(args.swapper_model):
        raise FileNotFoundError(
            f"swapper model not found: {args.swapper_model}\n"
            f"put inswapper_128.onnx into deep/models first"
        )

    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    if src is None:
        raise RuntimeError(f"failed to read source image: {args.source}")
    if tgt is None:
        raise RuntimeError(f"failed to read target image: {args.target}")

    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    src_faces = app.get(src)
    tgt_faces = app.get(tgt)
    src_face = pick_largest_face(src_faces)
    tgt_face = pick_largest_face(tgt_faces)
    if src_face is None:
        raise RuntimeError("no face detected in source image")
    if tgt_face is None:
        raise RuntimeError("no face detected in target image")

    if args.source_refs.strip():
        ref_paths = [s.strip() for s in args.source_refs.split(",") if s.strip()]
        ref_paths = [p for p in ref_paths if os.path.exists(p) and p.lower().endswith(".png")]
        merged = average_source_embedding(app, [args.source] + ref_paths)
        if merged is not None:
            try:
                src_face.normed_embedding = merged
                src_face.embedding = merged
            except Exception:
                pass

    work = tgt.copy()
    if args.mode == "fullface":
        work = fullface_blend(src, work, src_face, tgt_face)
    elif args.mode == "headpaste":
        work = headpaste_blend(src, work, src_face, tgt_face, head_strength=head_strength)
    elif args.mode == "headreplace":
        work = headreplace_blend(src, work, src_face, tgt_face)

    if args.mode == "headreplace":
        result = work
    else:
        swapper = get_model(args.swapper_model, providers=["CPUExecutionProvider"])
        work_faces = app.get(work)
        work_face = pick_largest_face(work_faces) if work_faces else None
        if work_face is None:
            work_face = tgt_face
        swapped = swapper.get(work, work_face, src_face, paste_back=True)
        if args.mode == "features":
            result = features_blend(swapped, tgt, tgt_face, strength=head_strength)
        else:
            result = swapped

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(args.output, result)
    if not ok:
        raise RuntimeError(f"failed to write output image: {args.output}")

    print(f"deep face swap done: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        raise SystemExit(1)
