import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    data = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 3.0],
        ]
    )
    mean = data.mean(axis=0)
    centered = data - mean
    cov = (centered.T @ centered) / len(data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    if v[1] < 0:
        v = -v

    t = np.linspace(-3.0, 3.0, 200)
    line = mean + np.outer(t, v)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[:, 0], data[:, 1], color="tab:blue", zorder=3)
    for idx, (x_val, y_val) in enumerate(data, start=1):
        ax.annotate(
            f"p{idx}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.plot(line[:, 0], line[:, 1], color="black", linewidth=1.5)
    ax.set_title("Step 7a: Data and first principal component")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pca_data_line.pdf")
    plt.close(fig)

    z = centered @ v
    proj = mean + np.outer(z, v)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(line[:, 0], line[:, 1], color="black", linewidth=1.5)
    ax.scatter(proj[:, 0], proj[:, 1], color="tab:orange", zorder=3)
    for idx, (x_val, y_val) in enumerate(proj, start=1):
        ax.annotate(
            f"p{idx}'",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.set_title("Step 7b: Projected points on PC1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pca_projected.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
