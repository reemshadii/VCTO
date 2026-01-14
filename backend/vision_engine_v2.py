import cv2
import mediapipe as mp
import numpy as np
from skimage import color

class FeatureExtractorV2:
    """Enhanced feature extraction with multi-zone sampling and outlier rejection"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_robust_color(self, image_lab, mask):
        """Filters out shadows, highlights, and noise using 15th-85th percentile."""
        pixels = image_lab[mask > 0]
        if len(pixels) == 0: 
            return np.array([50, 0, 0])  # Mid-grey fallback
        
        l_channel = pixels[:, 0]
        lower_pc = np.percentile(l_channel, 15)
        upper_pc = np.percentile(l_channel, 85)
        
        robust_pixels = pixels[(l_channel >= lower_pc) & (l_channel <= upper_pc)]
        
        if len(robust_pixels) == 0:
            return np.median(pixels, axis=0)
        
        return np.median(robust_pixels, axis=0)

    def get_skin_sampling_enhanced(self, image_lab, landmarks, h, w):
        """Sample from 9 strategic facial zones with weighted averaging"""
        zones = {
            'forehead_center': 10,
            'forehead_left': 54,
            'forehead_right': 284,
            'cheek_left_upper': 117,
            'cheek_left_lower': 187,
            'cheek_right_upper': 346,
            'cheek_right_lower': 411,
            'jaw_left': 177,
            'jaw_right': 401
        }
        
        skin_samples = {}
        for zone_name, idx in zones.items():
            pt = landmarks[idx]
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = int(pt.x * w), int(pt.y * h)
            
            # Use circular sampling instead of rectangular
            cv2.circle(mask, (cx, cy), 8, 255, -1)
            skin_samples[zone_name] = self.get_robust_color(image_lab, mask)
        
        # Calculate weighted average (center zones weighted more)
        weights = {
            'forehead_center': 2.0,
            'forehead_left': 1.5,
            'forehead_right': 1.5,
            'cheek_left_upper': 2.0,
            'cheek_right_upper': 2.0,
            'cheek_left_lower': 1.5,
            'cheek_right_lower': 1.5,
            'jaw_left': 1.0,
            'jaw_right': 1.0
        }
        
        weighted_sum = np.zeros(3)
        total_weight = 0
        
        for zone, sample in skin_samples.items():
            weight = weights[zone]
            weighted_sum += sample * weight
            total_weight += weight
        
        avg_skin = weighted_sum / total_weight
        
        return avg_skin, skin_samples

    def get_hair_logic_v2(self, image_lab, landmarks, h, w):
        """Multi-zone hair sampling with outlier rejection"""
        # Sample from 3 different hair regions
        hair_zones = [
            ([10, 151, 9], -25),      # Top forehead area
            ([338, 109], -20),         # Side areas
            ([108, 337], -30)          # Temple areas (deeper into hairline)
        ]
        
        samples = []
        for landmark_ids, y_offset in hair_zones:
            zone_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in landmark_ids:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(zone_mask, 
                             (int(pt.x * w), int(pt.y * h + y_offset)), 
                             12, 255, -1)
            
            if np.sum(zone_mask) > 0:
                color_sample = self.get_robust_color(image_lab, zone_mask)
                samples.append(color_sample)
        
        if len(samples) == 0:
            return np.array([50, 0, 0])
        
        # Remove outliers using median absolute deviation
        samples = np.array(samples)
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        
        # Keep samples within 2 MAD of median
        if np.any(mad > 0):
            valid_samples = samples[np.all(np.abs(samples - median) < 2 * (mad + 1e-6), axis=1)]
        else:
            valid_samples = samples
        
        return np.mean(valid_samples, axis=0) if len(valid_samples) > 0 else median

    def get_eyebrow_color_v2(self, image_lab, landmarks, h, w):
        """Sample both eyebrows for more robust hair color fallback"""
        eyebrow_zones = {
            'left_brow': [70, 63, 105, 66, 107],
            'right_brow': [336, 296, 334, 293, 300]
        }
        
        samples = []
        for zone_name, indices in eyebrow_zones.items():
            eb_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(eb_mask, (int(pt.x * w), int(pt.y * h)), 5, 255, -1)
            
            if np.sum(eb_mask) > 0:
                color_sample = self.get_robust_color(image_lab, eb_mask)
                samples.append(color_sample)
        
        return np.mean(samples, axis=0) if samples else np.array([50, 0, 0])

    def get_eye_color_enhanced(self, image_rgb, image_lab, landmarks, h, w):
        """Enhanced iris detection with multi-stage filtering"""
        # Sample BOTH irises for better accuracy
        iris_zones = [
            [468, 469, 470, 471, 472],  # Right iris
            [473, 474, 475, 476, 477]   # Left iris
        ]
        
        iris_samples = []
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        for iris_indices in iris_zones:
            iris_mask = np.zeros((h, w), dtype=np.uint8)
            valid_pts = []
            for i in iris_indices:
                if i < len(landmarks):
                    valid_pts.append((int(landmarks[i].x * w), int(landmarks[i].y * h)))
            
            if len(valid_pts) >= 3:
                iris_pts = np.array(valid_pts)
                cv2.fillConvexPoly(iris_mask, iris_pts, 255)
                
                # Multi-stage filtering
                # 1. Remove catchlights (very bright pixels)
                iris_mask[image_lab[:, :, 0] > 85] = 0
                
                # 2. Remove low saturation (sclera/reflections)
                iris_mask[hsv[:, :, 1] < 25] = 0
                
                # 3. Remove extreme values
                iris_mask[hsv[:, :, 2] < 20] = 0
                iris_mask[hsv[:, :, 2] > 250] = 0
                
                if np.sum(iris_mask) > 10:  # Ensure we have enough pixels
                    color_sample = self.get_robust_color(image_lab, iris_mask)
                    iris_samples.append(color_sample)
        
        return np.mean(iris_samples, axis=0) if iris_samples else np.array([50, 0, 0])

    def apply_white_balance_correction(self, image_rgb):
        """Correct for warm color casts in images - AGGRESSIVE for studio photos"""
        # Calculate average color cast
        avg_b = np.mean(image_rgb[:, :, 2])  # Blue channel
        avg_r = np.mean(image_rgb[:, :, 0])  # Red channel
        avg_g = np.mean(image_rgb[:, :, 1])  # Green channel
        
        # If image is warm (more red than blue), cool it down
        if avg_r > avg_b + 5:  # Changed from 10 to 5 - more sensitive
            # Apply stronger cooling for studio photos
            correction_factor = min((avg_r - avg_b) / 200.0, 0.25)  # Max 25% correction (was 15%)
            image_rgb = image_rgb.copy().astype(np.float32)
            
            # Boost blue, reduce red
            image_rgb[:, :, 2] = np.clip(image_rgb[:, :, 2] * (1 + correction_factor * 1.2), 0, 255)
            image_rgb[:, :, 0] = np.clip(image_rgb[:, :, 0] * (1 - correction_factor * 0.7), 0, 255)
            
            return image_rgb.astype(np.uint8)
        
        return image_rgb
    
    def extract_features_from_memory(self, img, is_dyed=False):
        """Main extraction method with enhanced sampling"""
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply white balance correction for warm-tinted images
        image_rgb = self.apply_white_balance_correction(image_rgb)
        
        image_lab = color.rgb2lab(image_rgb / 255.0)
        h, w, _ = img.shape
        
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected")
        
        landmarks = results.multi_face_landmarks[0].landmark

        # --- ENHANCED SKIN SAMPLING ---
        final_skin_lab, skin_samples = self.get_skin_sampling_enhanced(
            image_lab, landmarks, h, w
        )

        # --- ENHANCED EYE SAMPLING ---
        eye_color_lab = self.get_eye_color_enhanced(
            image_rgb, image_lab, landmarks, h, w
        )

        # --- ENHANCED HAIR/EYEBROW SAMPLING ---
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