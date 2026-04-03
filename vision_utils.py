import cv2
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, dropout_rate=0.4, dense_units=128): 
        super(ConvNet, self).__init__()
        
        def make_block(in_c, out_c, double_conv=True):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()]
            if double_conv:
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()]
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.conv1 = make_block(1, 32, double_conv=False)
        self.conv2 = make_block(32, 64)
        self.conv3 = make_block(64, 128)
        self.conv4 = make_block(128, 256)
        self.conv5 = make_block(256, 512)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, dense_units), nn.ReLU(), 
            nn.BatchNorm1d(dense_units), nn.Dropout(dropout_rate), 
            nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x); x = self.conv_final(x)
        x = self.adaptive_pool(x); x = self.flatten(x); x = self.classifier(x)
        return x

def load_vision_model(model_path):
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_eye(eye_img, img_size=128):
    try:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        tensor = torch.tensor(resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(device)
    except: 
        return None

def get_eye_roi(frame, landmarks, eye_indices, padding=5):
    h, w, _ = frame.shape
    x_coords = [int(landmarks[idx].x * w) for idx in eye_indices]
    y_coords = [int(landmarks[idx].y * h) for idx in eye_indices]
    min_x, max_x = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
    min_y, max_y = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)
    return frame[min_y:max_y, min_x:max_x]