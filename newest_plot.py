import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ----------------------------------------------------------------------
# 1. File paths (using your existing PNGs)
# ----------------------------------------------------------------------
wing_imgs = [
    "artifacts/eval_results/qual_wing_0.png",
    "artifacts/eval_results/qual_wing_1.png",
    "artifacts/eval_results/qual_wing_2.png"
]

rand_imgs = [
    "artifacts/eval_results/qual_rand_0.png",
    "artifacts/eval_results/qual_rand_1.png",
    "artifacts/eval_results/qual_rand_2.png"
]

# ----------------------------------------------------------------------
# 2. Paper-style aesthetic settings
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
})

fig, axs = plt.subplots(2, 3, figsize=(10, 5))

# ----------------------------------------------------------------------
# 3. Top row (validation reconstructions)
# ----------------------------------------------------------------------
for i, path in enumerate(wing_imgs):
    img = mpimg.imread(path)
    axs[0, i].imshow(img, cmap="gray")
    axs[0, i].set_title(f"Validation {i}")
    axs[0, i].axis("off")

# ----------------------------------------------------------------------
# 4. Bottom row (test-set inpainting)
# ----------------------------------------------------------------------
for i, path in enumerate(rand_imgs):
    img = mpimg.imread(path)
    axs[1, i].imshow(img, cmap="gray")
    axs[1, i].set_title(f"Inpainting {i}")
    axs[1, i].axis("off")

# ----------------------------------------------------------------------
# 5. Final layout
# ----------------------------------------------------------------------
plt.tight_layout()
plt.savefig("qualitative_figure.png", dpi=300, bbox_inches="tight")
plt.show()
