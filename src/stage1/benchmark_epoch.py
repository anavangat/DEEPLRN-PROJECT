"""This prints a real per-epoch cost and extrapolates the whole project so you can 
adjust imgsz or batch before burning GPU hours.

    python -m src.stage1.benchmark_epoch --data data/prepared/rgb/data1.yaml
"""
import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3, help="time this many, report the mean")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This script is A100-only by design.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    from ultralytics import YOLO

    t0 = time.time()
    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=8, seed=0, deterministic=True, val=True,
        project="runs/_benchmark", name="probe", exist_ok=True, verbose=False,
        plots=False,
    )
    elapsed = time.time() - t0
    per_epoch = elapsed / args.epochs
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n--- BENCHMARK ---")
    print(f"total {elapsed/60:.1f} min for {args.epochs} epochs")
    print(f"per epoch: {per_epoch:.1f} s   peak GPU mem: {peak:.1f} GB")

    tune_ep, final_ep, seeds = 10, 100, 5
    tune_h = 15 * tune_ep * per_epoch / 3600
    final_h = seeds * final_ep * per_epoch / 3600
    print(f"\nEXTRAPOLATION (Stage 1 only)")
    print(f"  Optuna 15 trials x {tune_ep} ep : {tune_h:.1f} h")
    print(f"  Final {seeds} seeds x {final_ep} ep : {final_h:.1f} h")
    print(f"  Stage 1 subtotal              : {tune_h + final_h:.1f} h")
    if tune_h + final_h > 6:
        print("\n  >6 h. Cut cost with: --imgsz 416, fewer final epochs, or "
              "epochs=8 in tuning.")


if __name__ == "__main__":
    main()
