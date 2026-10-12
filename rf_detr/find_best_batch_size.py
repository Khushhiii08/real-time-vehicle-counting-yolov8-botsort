# Place this near the top of the script, after CUDA is available
import torch

device = torch.device("cuda", 0)
torch.cuda.memory.set_per_process_memory_fraction(0.9, device)  # 90% cap [web:40]

def peak_bytes_this_step(step_fn):
    torch.cuda.synchronize(device)
    torch.cuda.memory.reset_peak_memory_stats(device)  # reset peak [web:44]
    step_fn()  # run exactly one forward+backward (or a tiny fixed number of steps)
    torch.cuda.synchronize(device)
    return torch.cuda.memory.max_memory_allocated(device)  # bytes [web:31]

def try_one_step(model_step):
    try:
        bytes_peak = peak_bytes_this_step(model_step)
        return True, bytes_peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, None  # too big [web:21]

# Example step function maker for RF‑DETR: create a closure that runs ONE train step at a given batch size.
# If the high-level API lacks a single-step mode, approximate with a tiny epoch over a very small sampler.
# Ensure AMP is enabled inside to reflect real memory use [web:13].
def make_one_step_fn(rfdetr_model, dataset_dir, bs, resolution, lr):
    def step():
        # Configure a tiny run (e.g., epochs=1 with minimal data) if supported;
        # otherwise call the library's "one step" or a custom small loop if available.
        # The key is: do exactly one forward+backward with bs, resolution, AMP.
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # AMP [web:13]
            # This call should perform exactly one train step; if not available in the API,
            # emulate a single step using the library’s lower-level hooks.
            rfdetr_model.train(
                dataset_dir=dataset_dir,
                epochs=1,
                batch_size=bs,
                grad_accum_steps=1,
                lr=lr,
                resolution=resolution,
                output_dir="tmp_probe",
                num_workers=0,
                device="cuda",
                early_stopping=False
            )
    return step

def find_max_batch_size(rfdetr_model, dataset_dir, resolution, lr=5e-3, bs_start=1, bs_max=64):
    # Exponential search to get an upper bound
    bs = bs_start
    last_ok = 0
    while bs <= bs_max:
        ok, _ = try_one_step(make_one_step_fn(rfdetr_model, dataset_dir, bs, resolution, lr))
        if not ok:
            break
        last_ok = bs
        bs *= 2

    if last_ok == 0:
        return 1  # fallback to 1 if even bs=1 fails under the cap [web:40][web:21]

    lo, hi = last_ok, min(bs, bs_max)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, _ = try_one_step(make_one_step_fn(rfdetr_model, dataset_dir, mid, resolution, lr))
        if ok:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
