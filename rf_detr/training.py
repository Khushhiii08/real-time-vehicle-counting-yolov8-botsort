import torch
import os
from rfdetr import RFDETRBase

if __name__ == '__main__':
    # Optional: global performance knobs for Torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # For Ada GPUs like 4060

    DATASET_PATH = r"\Khushi\RF-DETR\IDD_Detection_rfdetr\IDD_Detection_rfdetr"
    OUTPUT_PATH = "rfdetr_optimized"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Instantiate with pretrained weights if needed, e.g.:
    # model = RFDETRBase(pretrain_weights="path_to_weights.pt") 
    model = RFDETRBase()

    # Recommended dataloader settings for fast CPU feeding
    num_workers = 2            # Tune: try 4, 8, 12 as appropriate
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4        # Only if your train function supports it

    # Set device, no need to call model.to(...)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train(
        dataset_dir=DATASET_PATH,
        epochs=200,
        batch_size=4,   # Integer, e.g., batch_size=4
        grad_accum_steps=4,        # Integer, for effective batch
        lr=0.001,
        resolution=504,                              # Lowering to speed up epoch
        output_dir=OUTPUT_PATH,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,           # If supported by API
        seed=42,
        device=device,                               # << only this, no .to(...)!
        weight_decay=0.0005,
        early_stopping=True,
        clip_max_norm=0.1,
        early_stopping_patience=50,
        use_amp=True,                              # If train() accepts it, speeds up training
        resume="C:\\Khushi\\RF-DETR\\rfdetr_optimized\\checkpoint.pth"
    )