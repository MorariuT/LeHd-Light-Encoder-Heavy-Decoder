from LeHd import *
from Helpers import *
from PerceptualLoss import *
import time

device = torch.device("mps")
H, W, B = 256, 256, 1
print(f"Device: {device}\n{'='*60}")

enc = LeHdEncoder().to(device).eval()
dec = LeHdDecoder().to(device).eval()
dummy = torch.randn(B, 3, H, W, device=device)

print(f"\n{'Mode':<6} {'Payload (KB)':<14} {'Output'}")
print("-" * 45)

with torch.no_grad():
    for mode in range(6):
        payload = enc(dummy, mode=mode)
        recon = dec(payload)
        est = estimate_payload_bytes(B, H, W, mode)
        print(f"  {mode}    {est['total_kb']:>8.1f} KB     {tuple(recon.shape)}")
        assert recon.shape == (B, 3, H, W)
        assert 0.0 <= recon.min() and recon.max() <= 1.0

print("\n── Encoder throughput (mode=0, 200 frames) ──")

for _ in range(10): enc(dummy, mode=0)

if device.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.no_grad():
    for _ in range(200): enc(dummy, mode=0)

if device.type == "cuda": torch.cuda.synchronize()
fps = 200 / (time.perf_counter() - t0)

print(f"  {fps:.1f} FPS")

enc_p = sum(p.numel() for p in enc.parameters()) / 1e6
dec_p = sum(p.numel() for p in dec.parameters()) / 1e6

print(f"\nEncoder : {enc_p:.2f} M params  (device)")
print(f"Decoder : {dec_p:.2f} M params  (server)")
print(f"Total   : {enc_p + dec_p:.2f} M params")
print("\nAll checks passed.")

train_on_video(video_path="../datasets/movie3.mp4", model_path="../models/model_v2.pth")