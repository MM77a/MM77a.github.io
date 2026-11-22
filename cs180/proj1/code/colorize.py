from pathlib import Path

import cv2 as cv
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def align(source, reference, max_shift=15, crop_ratio=0.2):
    """
    Align source to reference with exhaustive L2 search.

    Returns the aligned source and its displacement as (x, y).
    """
    height, width = source.shape

    # Exclude strong plate borders so they do not dominate the score.
    border_y = max(max_shift + 1, int(height * crop_ratio))
    border_x = max(max_shift + 1, int(width * crop_ratio))

    reference_center = reference[
        border_y:-border_y,
        border_x:-border_x
    ]

    best_score = np.inf
    best_dx = 0
    best_dy = 0

    # Test every translation in the specified search window.
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(
                source,
                shift=(dy, dx),
                axis=(0, 1)
            )

            shifted_center = shifted[
                border_y:-border_y,
                border_x:-border_x
            ]

            # MSE has the same minimizer as squared L2 distance.
            score = np.mean(
                (shifted_center - reference_center) ** 2
            )

            if score < best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    # Apply the best displacement to the full-resolution channel.
    aligned = np.roll(
        source,
        shift=(best_dy, best_dx),
        axis=(0, 1)
    )

    return aligned, (best_dx, best_dy)

def pyramid_align(
    source,
    reference,
    coarse_size=250,
    coarse_radius=15,
    refine_radius=3,
    crop_ratio=0.2
):
    """
    Align images with a recursive coarse-to-fine pyramid.

    The displacement is estimated at the smallest scale, doubled at
    each finer scale, and locally refined.
    """

    # At the coarsest level, a full search is computationally affordable.
    if max(source.shape) <= coarse_size:
        return align(
            source,
            reference,
            max_shift=coarse_radius,
            crop_ratio=crop_ratio
        )

    # Construct the next pyramid level by downsampling both channels.
    source_small = cv.resize(
        source,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv.INTER_AREA
    )
    reference_small = cv.resize(
        reference,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv.INTER_AREA
    )

    # Recursively estimate displacement at the coarser scale.
    _, (small_dx, small_dy) = pyramid_align(
        source_small,
        reference_small,
        coarse_size=coarse_size,
        coarse_radius=coarse_radius,
        refine_radius=refine_radius,
        crop_ratio=crop_ratio
    )

    # Convert the coarse displacement to the current resolution.
    estimated_dx = small_dx * 2
    estimated_dy = small_dy * 2

    source_shifted = np.roll(
        source,
        shift=(estimated_dy, estimated_dx),
        axis=(0, 1)
    )

    # Refine only near the coarse estimate instead of searching globally.
    _, (correction_dx, correction_dy) = align(
        source_shifted,
        reference,
        max_shift=refine_radius,
        crop_ratio=crop_ratio
    )

    final_dx = estimated_dx + correction_dx
    final_dy = estimated_dy + correction_dy

    aligned = np.roll(
        source,
        shift=(final_dy, final_dx),
        axis=(0, 1)
    )

    return aligned, (final_dx, final_dy)

def edge_features(image):
    """Compute Sobel gradient magnitude for feature-based alignment."""
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)

    return cv.magnitude(grad_x, grad_y)

def process_image(filename, mode="pyramid"):
    """Load, align, colorize, and save one stacked glass-plate image."""

    # Load without downcasting 16-bit TIFFs, then normalize to [0, 1].
    image_path = DATA_DIR / filename
    im = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)

    if im is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    max_value = np.iinfo(im.dtype).max
    im = im.astype(np.float32) / max_value

    # Split the vertically stacked plate in its original B, G, R order.
    height = im.shape[0] // 3

    b = im[:height, :]
    g = im[height:2 * height, :]
    r = im[2 * height:3 * height, :]

    # Align the green and red channels to the fixed blue reference.
    if mode == "single_scale":
        ag, g_offset = align(g, b)
        ar, r_offset = align(r, b)

    elif mode == "pyramid":
        ag, g_offset = pyramid_align(g, b)
        ar, r_offset = pyramid_align(r, b)

    elif mode == "edges":
        # Estimate offsets from Sobel features, then apply them to the
        # original intensity channels to preserve the final colors.
        b_features = edge_features(b)
        g_features = edge_features(g)
        r_features = edge_features(r)

        _, g_offset = pyramid_align(g_features, b_features)
        _, r_offset = pyramid_align(r_features, b_features)

        ag = np.roll(
            g,
            shift=(g_offset[1], g_offset[0]),
            axis=(0, 1)
        )
        ar = np.roll(
            r,
            shift=(r_offset[1], r_offset[0]),
            axis=(0, 1)
        )

    else:
        raise ValueError(
            "Mode must be 'single_scale', 'pyramid', or 'edges'"
        )

    # Stack in RGB order for viewing.
    im_out = np.dstack([ar, ag, b])

    # OpenCV writes images in BGR order, so convert before saving.
    out_uint8 = np.clip(im_out * 255, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_{mode}.jpg"

    if not cv.imwrite(str(output_path), out_bgr):
        raise IOError(f"Could not save {output_path}")

    # Report displacements in the project-required (x, y) convention.
    print(f"{filename} [{mode}]")
    print(f"  G offset (x, y): {g_offset}")
    print(f"  R offset (x, y): {r_offset}")

    return g_offset, r_offset


LOW_RES_FILES = [
    "cathedral.jpg",
    "monastery.jpg",
    "tobolsk.jpg",
]

OFFICIAL_FILES = [
    "cathedral.jpg",
    "monastery.jpg",
    "tobolsk.jpg",
    "church.tif",
    "emir.tif",
    "harvesters.tif",
    "icon.tif",
    "ilemselga.tif",
    "melons.tif",
    "religous_painting.tif",
    "self_portrait.tif",
    "siren.tif",
    "three_generations.tif",
    "wharf.tif",
]

CUSTOM_FILES = [
    "sorochei_dam.tif",
    "v_malorossii.tif",
    "makhrovye_maki.tif",
]


if __name__ == "__main__":
    # Required single-scale results
    for filename in LOW_RES_FILES:
        process_image(filename, mode="single_scale")

    # Required raw-pixel pyramid results
    for filename in OFFICIAL_FILES + CUSTOM_FILES:
        process_image(filename, mode="pyramid")

    # Better-feature result for Emir
    process_image("emir.tif", mode="edges")