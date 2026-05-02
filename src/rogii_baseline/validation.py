"""Validation command line entry point."""

from __future__ import annotations

import argparse

from .baseline import MODEL_KINDS, evaluate_baseline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run masked train-time validation.")
    parser.add_argument("--data-dir", default=None, help="Competition data directory.")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default="geometry")
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-val-wells", type=int, default=100)
    parser.add_argument("--ps-fraction", type=float, default=0.30)
    parser.add_argument("--mask-strategy", choices=["actual", "artificial"], default="artificial")
    parser.add_argument("--pf-particles", type=int, default=500, help="Particles per PF seed for pf/physical_pf.")
    parser.add_argument("--pf-seeds", type=int, default=32, help="PF seed count for validation.")
    parser.add_argument("--pf-likelihood-scale", type=float, default=5.0, help="Scale for PF seed likelihood weighting.")
    parser.add_argument("--beam-blend-weight", type=float, default=0.0, help="Optional beam-search weight for PF validation.")
    args = parser.parse_args(argv)
    result = evaluate_baseline(
        data_dir=args.data_dir,
        model_kind=args.model_kind,
        max_train_rows=args.max_train_rows,
        max_val_wells=args.max_val_wells,
        ps_fraction=args.ps_fraction,
        mask_strategy=args.mask_strategy,
        pf_n_particles=args.pf_particles,
        pf_n_seeds=args.pf_seeds,
        pf_likelihood_scale=args.pf_likelihood_scale,
        beam_blend_weight=args.beam_blend_weight,
    )
    if result.empty:
        print("No validation rows scored.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
