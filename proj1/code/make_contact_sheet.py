from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"

names = [
    "cathedral", "monastery", "tobolsk", "church",
    "emir", "harvesters", "icon", "ilemselga",
    "melons", "religous_painting", "self_portrait",
    "siren", "three_generations", "wharf",
]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for ax, name in zip(axes, names):
    filename = (
        "emir_edges.jpg"
        if name == "emir"
        else f"{name}_pyramid.jpg"
    )

    image = plt.imread(OUTPUT_DIR / filename)

    # Crop borders only for easier visual inspection
    h, w = image.shape[:2]
    image = image[int(h * 0.04):int(h * 0.96),
                  int(w * 0.04):int(w * 0.96)]

    ax.imshow(image)
    ax.set_title(name.replace("_", " "))
    ax.axis("off")

for ax in axes[len(names):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "contact_sheet.jpg", dpi=150)