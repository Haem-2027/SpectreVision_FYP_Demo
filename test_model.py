from ultralytics import YOLO

for name in ["box_loading", "helmet_best" ]:
    model = YOLO("models/" + name + ".pt")
    print(name, "can detect:", model.names)

