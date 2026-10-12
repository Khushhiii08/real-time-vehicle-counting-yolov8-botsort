if __name__ == "__main__":
    from ultralytics import YOLO
    import torch
    import ultralytics

    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")

    model = YOLO('yolov8m.pt')
    model.model.nc = 15  # number of classes

    try:
        model.train(
            data='/Users/jolliojelly/Downloads/PROJECTS/VehicleCounting/IDD_Detection_yolo/dataset.yaml',
            epochs=200,
            imgsz=416,
            batch=2,
            optimizer='AdamW',
            seed=42,
            device='mps',
            workers=2,
            name='yolov8m_optimized',
            lr0=0.004,
            lrf=0.05,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            scale=0.9,
            shear=0.3,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.1,
            max_det=300,
            conf=0.25
        )
    except Exception as e:
        print("Training stopped due to error:", e)

    print("Training complete.")

    # Save the model
    model.save('yolov8trained.pt')