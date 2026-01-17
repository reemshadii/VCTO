import cv2
import numpy as np
from skimage import color
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

class FeatureExtractorV2:
    
    def __init__(self):
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("📥 Downloading face detection model...")
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model downloaded!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_robust_color(self, image_lab, mask):
        pixels = image_lab[mask > 0]
        if len(pixels) == 0: 
            return np.array([50, 0, 0])
        
        l_channel = pixels[:, 0]
        lower_pc = np.percentile(l_channel, 15)
        upper_pc = np.percentile(l_channel, 85)
        
        robust_pixels = pixels[(l_channel >= lower_pc) & (l_channel <= upper_pc)]
        
        if len(robust_pixels) == 0:
            return np.median(pixels, axis=0)
        
        return np.median(robust_pixels, axis=0)

    def get_skin_sampling_enhanced(self, image_lab, landmarks, h, w):
        zones = {
            'forehead_center': 10, 'forehead_left': 54, 'forehead_right': 284,
            'cheek_left_upper': 117, 'cheek_left_lower': 187,
            'cheek_right_upper': 346, 'cheek_right_lower': 411,
            'jaw_left': 177, 'jaw_right': 401
        }
        
        skin_samples = {}
        for zone_name, idx in zones.items():
            if idx < len(landmarks):
                pt = landmarks[idx]
                mask = np.zeros((h, w), dtype=np.uint8)
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(mask, (cx, cy), 8, 255, -1)
                skin_samples[zone_name] = self.get_robust_color(image_lab, mask)
        
        weights = {
            'forehead_center': 2.0, 'forehead_left': 1.5, 'forehead_right': 1.5,
            'cheek_left_upper': 2.0, 'cheek_right_upper': 2.0,
            'cheek_left_lower': 1.5, 'cheek_right_lower': 1.5,
            'jaw_left': 1.0, 'jaw_right': 1.0
        }
        
        weighted_sum = np.zeros(3)
        total_weight = 0
        
        for zone, sample in skin_samples.items():
            weight = weights.get(zone, 1.0)
            weighted_sum += sample * weight
            total_weight += weight
        
        return weighted_sum / total_weight, skin_samples

    def get_hair_logic_v2(self, image_lab, landmarks, h, w):
        hair_zones = [([10, 151, 9], -25), ([338, 109], -20), ([108, 337], -30)]
        
        samples = []
        for landmark_ids, y_offset in hair_zones:
            zone_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in landmark_ids:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(zone_mask, (int(pt.x * w), int(pt.y * h + y_offset)), 12, 255, -1)
            
            if np.sum(zone_mask) > 0:
                samples.append(self.get_robust_color(image_lab, zone_mask))
        
        if len(samples) == 0:
            return np.array([50, 0, 0])
        
        samples = np.array(samples)
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        
        if np.any(mad > 0):
            valid_samples = samples[np.all(np.abs(samples - median) < 2 * (mad + 1e-6), axis=1)]
        else:
            valid_samples = samples
        
        return np.mean(valid_samples, axis=0) if len(valid_samples) > 0 else median

    def get_eyebrow_color_v2(self, image_lab, landmarks, h, w):
        eyebrow_zones = {
            'left_brow': [70, 63, 105, 66, 107],
            'right_brow': [336, 296, 334, 293, 300]
        }
        
        samples = []
        for indices in eyebrow_zones.values():
            eb_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(eb_mask, (int(pt.x * w), int(pt.y * h)), 5, 255, -1)
            
            if np.sum(eb_mask) > 0:
                samples.append(self.get_robust_color(image_lab, eb_mask))
        
        return np.mean(samples, axis=0) if samples else np.array([50, 0, 0])

    def get_eye_color_enhanced(self, image_rgb, image_lab, landmarks, h, w):
        iris_zones = [[468, 469, 470, 471, 472], [473, 474, 475, 476, 477]]
        iris_samples = []
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        for iris_indices in iris_zones:
            iris_mask = np.zeros((h, w), dtype=np.uint8)
            valid_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                        for i in iris_indices if i < len(landmarks)]
            
            if len(valid_pts) >= 3:
                cv2.fillConvexPoly(iris_mask, np.array(valid_pts), 255)
                iris_mask[image_lab[:, :, 0] > 85] = 0
                iris_mask[hsv[:, :, 1] < 25] = 0
                iris_mask[hsv[:, :, 2] < 20] = 0
                iris_mask[hsv[:, :, 2] > 250] = 0
                
                if np.sum(iris_mask) > 10:
                    iris_samples.append(self.get_robust_color(image_lab, iris_mask))
        
        return np.mean(iris_samples, axis=0) if iris_samples else np.array([50, 0, 0])

    def apply_white_balance_correction(self, image_rgb):
        avg_b = np.mean(image_rgb[:, :, 2])
        avg_r = np.mean(image_rgb[:, :, 0])
        
        if avg_r > avg_b + 5:
            correction_factor = min((avg_r - avg_b) / 200.0, 0.25)
            image_rgb = image_rgb.copy().astype(np.float32)
            image_rgb[:, :, 2] = np.clip(image_rgb[:, :, 2] * (1 + correction_factor * 1.2), 0, 255)
            image_rgb[:, :, 0] = np.clip(image_rgb[:, :, 0] * (1 - correction_factor * 0.7), 0, 255)
            return image_rgb.astype(np.uint8)
        
        return image_rgb
    
    def extract_features_from_memory(self, img, is_dyed=False):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_rgb = self.apply_white_balance_correction(image_rgb)
        image_lab = color.rgb2lab(image_rgb / 255.0)
        h, w, _ = img.shape
        
        mp_image = python.Image(image_format=python.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.face_landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            raise ValueError("No face detected")
        
        landmarks = detection_result.face_landmarks[0]
        final_skin_lab, skin_samples = self.get_skin_sampling_enhanced(image_lab, landmarks, h, w)
        eye_color_lab = self.get_eye_color_enhanced(image_rgb, image_lab, landmarks, h, w)
        
        if is_dyed:
            hair_color_lab = self.get_eyebrow_color_v2(image_lab, landmarks, h, w)
        else:
            hair_color_lab = self.get_hair_logic_v2(image_lab, landmarks, h, w)
        
        return {
            'skin_lab': final_skin_lab,
            'skin_samples': skin_samples,
            'hair_lab': hair_color_lab,
            'eye_lab': eye_color_lab,
            'image_shape': (h, w)
        }
