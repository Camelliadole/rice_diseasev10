from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO(r'D:\Python code\V10\yolov10\ultralytics\cfg\models\v10\yolov10n.yaml')

# Load pretrained YOLO model
model = YOLO(r'D:\Python code\V10\weights\yolov10n.pt')

# Train the model using my custom dataset
data_path = r'D:\Python code\data.yaml'
model.train(data = data_path, epochs = 30)
model.val()
