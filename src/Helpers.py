from LeHd import *
from PerceptualLoss import *


def train_on_video(
    video_path: str  = "movie3.mp4",
    model_path: str  = "model_v2.pth",
    size: int        = 256,
    mode: int        = 0,           # bandwidth mode to train at
    device_str: str  = "mps",
    lr: float        = 1e-4,
    λ_mse: float     = 1.0,
    λ_perc: float    = 0.5,         # perceptual loss weight
    λ_freq: float    = 0.2,        # frequency loss weight
    save_every: int  = 500,         # also save every N frames
    save_frames: list = None,
) -> None:
    """
    Train LeHd on a local video file.

    Fixes vs v1
    ───────────
    • BGR→RGB: cv2 reads BGR; we convert to RGB before any tensor operations
      so the model and loss both operate in RGB colour space.
    • mode is passed explicitly to model.forward().
    • Perceptual + frequency loss replaces pure MSE.
    • Gradient clipping (norm ≤ 1.0).
    • model.train() wraps the backward pass; model.eval() wraps display inference.
    • Periodic saves every `save_every` frames, not just on ESC.
    """
    if save_frames is None:
        save_frames = [1, 1300, 1400, 1500, 1600, 1700]

    device = torch.device(device_str)

    # ── Model ──────────────────────────────────────────────────────────────
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
        print(f"Loaded saved model from {model_path}")
    except Exception:
        model = LeHd(pretrained=True).to(device)
        print("No saved model found, starting fresh.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()
    perc_fn = PerceptualLoss(device)

    # ── Video ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frame_itr = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_itr += 1

        # ── BGR → RGB, resize, to tensor ──────────────────────────────────
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp_np = cv2.resize(frame_rgb, (size, size))                # HWC uint8 RGB
        inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).float() / 255.0
        inp_t = inp_t.unsqueeze(0).to(device)                      # 1×3×H×W

        # ── Training step ─────────────────────────────────────────────────
        model.train()
        out = model(inp_t, mode=mode)

        loss_mse = mse_fn(out, inp_t)
        loss_perc = perc_fn(out, inp_t)
        loss_freq = frequency_loss(out, inp_t)
        loss = λ_mse * loss_mse + λ_perc * loss_perc + λ_freq * loss_freq

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # ── Display inference (eval, no_grad) ─────────────────────────────
        model.eval()
        with torch.no_grad():
            decoded_t = model(inp_t, mode=mode)

        decoded_np = (decoded_t[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Display both as BGR for cv2
        orig_bgr = cv2.cvtColor(inp_np, cv2.COLOR_RGB2BGR)
        decoded_bgr = cv2.cvtColor(decoded_np, cv2.COLOR_RGB2BGR)

        bar = np.zeros((4, size, 3), dtype=np.uint8)
        combined = np.vstack([orig_bgr, bar, decoded_bgr])

        # Overlay loss + mode info
        label = f"Mode {mode} | MSE {loss_mse.item():.4f} | Perc {loss_perc.item():.4f} | frame {frame_itr}"
        cv2.putText(combined, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)

        cv2.imshow("LeHd v2  [top: original | bottom: decoded]", combined)

        # ── Save frames ───────────────────────────────────────────────────
        if frame_itr in save_frames:
            out_name = f"../screenshots/frame_{frame_itr}_mode{mode}.jpg"
            cv2.imwrite(out_name, decoded_bgr)
            print(f"Saved {out_name}")

        if frame_itr % save_every == 0:
            torch.save(model, model_path)
            print(f"[frame {frame_itr}] checkpoint saved → {model_path}  loss={loss.item():.5f}")

        key = cv2.waitKey(1)
        if key == 27:
            print("Saving model and exiting...")
            torch.save(model, model_path)
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# Payload estimator
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_payload_bytes(batch: int, h: int, w: int, mode: int, dtype_bytes: int = 4) -> dict:
    base_elems = batch * LeHdEncoder.BOTTLENECK_CH * (h // 64) * (w // 64)
    res_spatial = [(h // 32, w // 32), (h // 16, w // 16),
                   (h // 8,  w // 8),  (h // 4,  w // 4), (h // 2, w // 2)]
    res_elems = sum(batch * LeHdEncoder.RESIDUAL_CH * rh * rw for rh, rw in res_spatial[:mode])
    total = (base_elems + res_elems) * dtype_bytes
    return {"mode": mode, "base_kb": base_elems * dtype_bytes / 1024,
            "residuals_kb": res_elems * dtype_bytes / 1024, "total_kb": total / 1024}

