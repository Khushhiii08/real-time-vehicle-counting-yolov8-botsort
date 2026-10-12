if __name__ == "__main__":
    from ultralytics import YOLO
    import torch
    import ultralytics

    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")

    # Load YOLOv12 model
    model = YOLO('yolo12m.pt')
    model.model.nc = 15  # number of classes

    try:
        model.train(
            data='/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/dataset.yaml',
            epochs=300,                # Increase epochs for longer convergence window
            imgsz=960,                 # Larger images for more detail, if possible
            batch=16,
            patience=80,               # Larger patience for slower improvements
            optimizer='AdamW',         # Try 'SGD' in another run to compare
            workers=4,
            seed=42,
            pretrained=True,
            device='cuda',             # Use GPU ('cuda') for full performance
            name='yolo12m_optimized',
            lr0=0.003,                 # Lower initial LR for stable training
            lrf=0.05,                  # Lower final LR
            momentum=0.937,
            weight_decay=0.001,        # Slightly more regularization
            warmup_epochs=4.0,
            warmup_momentum=0.85,
            warmup_bias_lr=0.1,
            box=10.0,                  # Greater emphasis on box loss
            cls=1.2,                   # More focus on rare class detection
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.5,
            translate=0.15,
            scale=0.8,
            shear=0.6,
            perspective=0.0,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.2,                # Slightly stronger mosaic
            mixup=0.3,                 # Stronger mixup for more diversity
            copy_paste=0.2             # Higher copy-paste to up rare class visibility
        )
    except Exception as e:
        print("Training stopped due to error:", e)

    print("Training complete.")

    # Save the model
    model.save('yolo12trained.pt')
