"""SegmentService — detect and separate individual fibril instances.

Pipeline (v2.5):
  1. Zero capsid disk + scale-bar strip before Frangi.  Drives ~110 K zone
     pixels above threshold=0.05 (vs ~32 K on the natural image).
  2. Frangi(sigmas=(3,5,8), black_ridges=True), normalised to [0,1].
  3. Binary mask at threshold, restricted to annular zone [0.85r, 2.0r].
     Capsid mask and scale-bar excluded.  Remove isolated ≤7-px blobs.
  4. Skeletonize + 8-connected label.  Apply size ≥ min_fibril_px and
     anchoring (closest pixel ≤ anchor_band_px from capsid surface).
  5. ANGULAR MERGE: group surviving fragments that share the same radial
     direction (angle difference ≤ merge_angle_deg) and whose radial ranges
     are within merge_gap_px of each other.  Each group = one fibril.
  6. Within each group, connect adjacent sub-components in radial order with
     straight line segments → continuous stroke from capsid outward.
  7. Dilate merged skeleton for rendering.
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.draw import line as draw_line_pixels
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

from hairpainter.utils.types import CapsidResult, FibrilInstance, SegmentResult

_STRUCT_8 = np.ones((3, 3), dtype=int)


class SegmentService:
    def __init__(
        self,
        frangi_sigmas: tuple[float, ...] = (3, 5, 8),
        frangi_threshold: float = 0.05,
        min_fibril_px: int = 15,
        dilation_radius: int = 3,
        anchor_band_px: int = 150,
        merge_angle_deg: float = 0.8,
        merge_gap_px: int = 30,
        bg_sigma: float = 20.0,
        zone_inner_frac: float = 0.85,
        zone_outer_frac: float = 2.0,
        capsid_mask_frac: float = 1.0,
        extend_inward_to_frac: float = 0.0,
    ) -> None:
        self._sigmas = frangi_sigmas
        self._threshold = frangi_threshold
        self._min_px = min_fibril_px
        self._dilation_radius = dilation_radius
        self._anchor_band_px = anchor_band_px
        self._merge_angle_rad = merge_angle_deg * math.pi / 180.0
        self._merge_gap_px = merge_gap_px
        self._bg_sigma = bg_sigma
        # Zone / capsid-exclusion geometry (defaults reproduce v2.5 behaviour).
        #   zone_*_frac      : annular fibril zone [inner*r, outer*r]
        #   capsid_mask_frac : exclusion-disk radius as a fraction of r; <1.0
        #                      exposes the inner peri-capsid ring that holds many
        #                      GT fibril roots (GT radial median ~1.1r, p10 ~0.2r).
        #   extend_inward_to_frac : if >0, extend each fibril root radially inward
        #                      down to this fraction of r (anchors roots to the
        #                      capsid surface; boosts inner-ring recall).
        self._zone_inner_frac = zone_inner_frac
        self._zone_outer_frac = zone_outer_frac
        self._capsid_mask_frac = capsid_mask_frac
        self._extend_inward_to_frac = extend_inward_to_frac

    def segment(self, enhanced: np.ndarray, capsid: CapsidResult) -> SegmentResult:
        h, w = enhanced.shape
        cx, cy = capsid.center
        r = capsid.radius
        scale_bar_y = int(h * 0.95)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2).astype(
            np.float32
        )

        # Exclusion disk: by default the full capsid mask; capsid_mask_frac < 1.0
        # shrinks it to expose the inner peri-capsid ring.
        if self._capsid_mask_frac >= 1.0:
            excl = capsid.mask
        else:
            excl = dist_from_center <= r * self._capsid_mask_frac

        fibril_zone = (dist_from_center >= r * self._zone_inner_frac) & (
            dist_from_center <= r * self._zone_outer_frac
        )
        fibril_zone[scale_bar_y:, :] = False
        fibril_zone[excl] = False

        dist_from_surface = np.abs(dist_from_center - r)

        # 1. Zero capsid disk before Frangi — boosts zone coverage from ~32 K
        #    to ~110 K pixels above threshold without creating a ring component.
        working = enhanced.astype(np.float32) / 255.0
        working[excl] = 0.0
        working[scale_bar_y:, :] = 0.0
        vesselness = frangi(working, sigmas=self._sigmas, black_ridges=True)
        mx = vesselness.max()
        if mx > 0:
            vesselness = vesselness / mx

        # 2. Binary mask restricted to the fibril zone.
        binary = (vesselness >= self._threshold).astype(bool)
        binary[excl] = False
        binary[~fibril_zone] = False
        binary[scale_bar_y:, :] = False
        binary = remove_small_objects(binary, max_size=7)

        # 3. Skeletonize + 8-connected label.
        skeleton = skeletonize(binary)
        skel_labeled, n_raw = ndi.label(skeleton, structure=_STRUCT_8)
        if n_raw == 0:
            return SegmentResult(label_map=np.zeros((h, w), dtype=np.int32), fibrils=[])

        sizes = np.bincount(skel_labeled.ravel())
        sizes[0] = 0

        # 4. Collect survivors: size filter + anchoring.
        #    Use a loose minimum (5 px) here; the final min_fibril_px is applied
        #    to the MERGED group so small fragments can combine into a full fibril.
        survivors: list[tuple[int, float, float, float]] = []
        for lbl in range(1, n_raw + 1):
            if sizes[lbl] < 5:
                continue
            skel_i = skel_labeled == lbl
            if dist_from_surface[skel_i].min() > self._anchor_band_px:
                continue
            ys_i, xs_i = np.where(skel_i)
            theta = math.atan2(float(ys_i.mean()) - cy, float(xs_i.mean()) - cx)
            d_min = float(dist_from_center[skel_i].min())
            d_max = float(dist_from_center[skel_i].max())
            survivors.append((lbl, theta, d_min, d_max))

        if not survivors:
            return SegmentResult(label_map=np.zeros((h, w), dtype=np.int32), fibrils=[])

        # 5. Angular merge: Union-Find over survivors.
        n = len(survivors)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        two_pi = 2.0 * math.pi
        for i in range(n):
            _, theta_i, d_min_i, d_max_i = survivors[i]
            for j in range(i + 1, n):
                _, theta_j, d_min_j, d_max_j = survivors[j]
                # Angular difference with wrap-around
                d_theta = abs(theta_i - theta_j)
                if d_theta > math.pi:
                    d_theta = two_pi - d_theta
                if d_theta > self._merge_angle_rad:
                    continue
                # Radial gap: positive means ranges don't overlap
                gap = max(d_min_i, d_min_j) - min(d_max_i, d_max_j)
                if gap <= self._merge_gap_px:
                    ri, rj = _find(i), _find(j)
                    if ri != rj:
                        parent[ri] = rj

        # Group members by root
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[_find(i)].append(i)

        # 6. Build FibrilInstances from groups.
        dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._dilation_radius * 2 + 1, self._dilation_radius * 2 + 1),
        )

        label_map = np.zeros((h, w), dtype=np.int32)
        fibrils: list[FibrilInstance] = []
        new_id = 1

        for root, member_idxs in groups.items():
            # Union all member skeletons
            skel_group = np.zeros((h, w), dtype=bool)
            for idx in member_idxs:
                lbl = survivors[idx][0]
                skel_group |= skel_labeled == lbl

            # Connect sub-components with radial line segments
            skel_group = self._connect_radial_fragments(
                skel_group, dist_from_center, excl, scale_bar_y
            )

            # Optional: extend the innermost end radially inward to anchor the
            # fibril root to the capsid surface (boosts inner-ring recall).
            if self._extend_inward_to_frac > 0.0:
                skel_group = self._extend_inward(
                    skel_group, dist_from_center, cx, cy,
                    r * self._extend_inward_to_frac, excl, scale_bar_y,
                )

            # Final size filter on the merged skeleton
            if skel_group.sum() < self._min_px:
                continue

            length_px = self._corrected_skeleton_length(skel_group)

            mask_i = cv2.dilate(skel_group.astype(np.uint8), dil_kernel).astype(bool)
            mask_i[excl] = False
            mask_i[scale_bar_y:, :] = False

            label_map[mask_i] = new_id
            fibrils.append(
                FibrilInstance(
                    id=new_id,
                    mask=mask_i,
                    skeleton=skel_group,
                    length_px=length_px,
                )
            )
            new_id += 1

        return SegmentResult(label_map=label_map, fibrils=fibrils)

    # ------------------------------------------------------------------
    @staticmethod
    def _connect_radial_fragments(
        skel: np.ndarray,
        dist_from_center: np.ndarray,
        capsid_mask: np.ndarray,
        scale_bar_y: int,
    ) -> np.ndarray:
        """Connect disconnected sub-components within a fibril group.

        Sub-components are sorted by their minimum distance from the capsid
        centre and connected outer-end → inner-end with straight Bresenham
        line segments.  Lines are clipped to avoid entering the capsid or
        the scale-bar strip.
        """
        h, w = skel.shape
        comp_lab, n_comp = ndi.label(skel, structure=_STRUCT_8)
        if n_comp <= 1:
            return skel

        # For each sub-component find its inner (closest to capsid) and
        # outer (farthest) skeleton pixels.
        inner_pts: list[tuple[float, int, int]] = []  # (d_min, x, y)
        outer_pts: list[tuple[float, int, int]] = []  # (d_max, x, y)
        for lbl in range(1, n_comp + 1):
            comp = comp_lab == lbl
            ys, xs = np.where(comp)
            dists = dist_from_center[ys, xs]
            idx_in = int(np.argmin(dists))
            idx_out = int(np.argmax(dists))
            inner_pts.append((float(dists[idx_in]), int(xs[idx_in]), int(ys[idx_in])))
            outer_pts.append((float(dists[idx_out]), int(xs[idx_out]), int(ys[idx_out])))

        # Sort by minimum distance (innermost sub-component first)
        order = sorted(range(n_comp), key=lambda i: inner_pts[i][0])

        result = skel.copy()
        for k in range(len(order) - 1):
            i_inner = order[k]
            i_outer = order[k + 1]
            # Connect outer end of inner component to inner end of outer component
            _, x0, y0 = outer_pts[i_inner]
            _, x1, y1 = inner_pts[i_outer]
            rr, cc = draw_line_pixels(y0, x0, y1, x1)
            valid = (
                (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                & ~capsid_mask[rr, cc]
                & (rr < scale_bar_y)
            )
            result[rr[valid], cc[valid]] = True

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _extend_inward(
        skel: np.ndarray,
        dist_from_center: np.ndarray,
        cx: int,
        cy: int,
        target_radius: float,
        excl: np.ndarray,
        scale_bar_y: int,
    ) -> np.ndarray:
        """Extend the innermost skeleton pixel radially inward to target_radius.

        Anchors the fibril root toward the capsid surface so the inner peri-
        capsid ring (where many GT fibrils live) is covered.  The drawn segment
        is clipped against the exclusion disk and scale-bar strip.
        """
        ys, xs = np.where(skel)
        if ys.size == 0:
            return skel
        d = dist_from_center[ys, xs]
        i_in = int(np.argmin(d))
        if d[i_in] <= target_radius:
            return skel
        x0, y0 = int(xs[i_in]), int(ys[i_in])
        theta = math.atan2(y0 - cy, x0 - cx)
        x1 = int(round(cx + target_radius * math.cos(theta)))
        y1 = int(round(cy + target_radius * math.sin(theta)))
        h, w = skel.shape
        rr, cc = draw_line_pixels(y0, x0, y1, x1)
        valid = (
            (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            & ~excl[rr.clip(0, h - 1), cc.clip(0, w - 1)]
            & (rr < scale_bar_y)
        )
        result = skel.copy()
        result[rr[valid], cc[valid]] = True
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _polar_frangi(
        gray: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        max_r: int,
        sigmas: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> np.ndarray:
        """Polar-space Frangi (kept for diagnostics; not used in the main pipeline)."""
        h, w = gray.shape
        n_angles = max(720, int(2 * np.pi * max_r))
        polar = cv2.warpPolar(
            gray,
            dsize=(max_r, n_angles),
            center=(float(cx), float(cy)),
            maxRadius=float(max_r),
            flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
        )
        polar_f = polar.astype(np.float32) / 255.0
        vessel_polar = frangi(polar_f, sigmas=sigmas, black_ridges=True)
        r_margin = 8
        r_start = max(0, int(r * 0.85) + r_margin)
        r_end = min(max_r - 1, int(r * 2.0) - r_margin)
        masked = np.zeros_like(vessel_polar)
        if r_start < r_end:
            masked[:, r_start:r_end] = vessel_polar[:, r_start:r_end]
        mx = masked.max()
        if mx > 0:
            masked = masked / mx
        vessel_cart = cv2.warpPolar(
            masked.astype(np.float32),
            dsize=(w, h),
            center=(float(cx), float(cy)),
            maxRadius=float(max_r),
            flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        return vessel_cart.astype(np.float32)

    @staticmethod
    def _background_subtract(gray: np.ndarray, sigma: float) -> np.ndarray:
        """Rolling-ball background subtraction (diagnostic use only)."""
        gray_f = gray.astype(np.float32)
        background = gaussian_filter(gray_f, sigma=sigma)
        response = background - gray_f
        response = np.clip(response, 0.0, None)
        mx = response.max()
        if mx > 0:
            response = response / mx * 255.0
        return response.astype(np.uint8)

    @staticmethod
    def _corrected_skeleton_length(skeleton: np.ndarray) -> float:
        """Arc-length estimate: straight step = 1, diagonal = sqrt(2)."""
        h_nb = skeleton[:, :-1] & skeleton[:, 1:]
        v_nb = skeleton[:-1, :] & skeleton[1:, :]
        d1_nb = skeleton[:-1, :-1] & skeleton[1:, 1:]
        d2_nb = skeleton[:-1, 1:] & skeleton[1:, :-1]
        straight = int(h_nb.sum()) + int(v_nb.sum())
        diag = int(d1_nb.sum()) + int(d2_nb.sum())
        return float(straight + diag * np.sqrt(2))
