"""Auto-generate matplotlib charts from benchmark JSONL results."""

import argparse
import json
import math
from pathlib import Path


def plot_jsonl(file_path: Path) -> None:
    from logger import log

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("Matplotlib not installed. Skipping auto-charts (pip install matplotlib).")
        return

    if not file_path.exists():
        log(f"ERROR: File not found -> {file_path}")
        return

    # Parse the JSONL file
    benchmarks: dict[str, dict[str, list[float]]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("_type") == "result":
                    bench_name = data.get("benchmark")
                    variant = data.get("variant")
                    loss_curve = data.get("loss_curve", [])
                    final_loss = data.get("final_loss")

                    # Skip benchmarks that don't have a loss curve
                    if not loss_curve:
                        continue

                    # If final_loss differs from the last active training step (e.g. SWA/EMA eval),
                    # append it so the chart explicitly visualizes the independent evaluation drop.
                    if final_loss is not None and isinstance(final_loss, (int, float)):
                        if (
                            not math.isnan(final_loss)
                            and abs(final_loss - loss_curve[-1]) > 1e-6
                        ):
                            loss_curve.append(final_loss)

                    if bench_name not in benchmarks:
                        benchmarks[bench_name] = {}

                    benchmarks[bench_name][variant] = loss_curve
            except json.JSONDecodeError:
                continue

    if not benchmarks:
        log("No valid loss curves found to plot.")
        return

    # Create an output directory next to the JSONL file
    out_dir = file_path.parent / f"{file_path.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save a plot for each benchmark category
    log(f"Generating plots in {out_dir} ...")
    for bench_name, variants in benchmarks.items():
        plt.figure(figsize=(10, 6))

        # Plot each variant's curve
        for variant, curve in variants.items():
            # Use dashed lines for baselines/off states to make them stand out
            linestyle = "--" if "off" in variant or "baseline" in variant else "-"
            plt.plot(curve, label=variant, linewidth=2, linestyle=linestyle)

        plt.title(f"Benchmark: {bench_name.title()}", fontsize=14, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Cross-Entropy Loss", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()

        # Save to disk
        out_file = out_dir / f"{bench_name}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        log(f"  ✓ Saved {bench_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot JSONL benchmark results."
    )
    parser.add_argument("input_file", type=str, help="Path to the .jsonl file to plot")
    args = parser.parse_args()

    # We redefine log as print when executing directly
    import builtins
    import logger

    logger.log = builtins.print

    plot_jsonl(Path(args.input_file))
