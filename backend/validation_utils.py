import cv2
import numpy as np
import json
from typing import Dict, List, Tuple

def detect_environmental_issues_v2(skin_samples, image_rgb, skin_lab):
    """Comprehensive image quality validation"""
    issues = []
    h, w = image_rgb.shape[:2]
    
    # 1. Lighting uniformity check
    l_values = [sample[0] for sample in skin_samples.values()]
    avg_l = np.mean(l_values)
    std_l = np.std(l_values)
    
    if avg_l < 25:
        issues.append(("CRITICAL", "Image is too dark. Please use better lighting."))
    elif avg_l < 35:
        issues.append(("WARNING", "Lighting is somewhat dark. Results may be less accurate."))
    elif avg_l > 85:
        issues.append(("WARNING", "Image is overexposed. Results may be less accurate."))
    
    if std_l > 12:
        issues.append(("WARNING", "Uneven lighting detected across face."))
    
    # 2. Color cast detection
    a_values = [sample[1] for sample in skin_samples.values()]
    b_values = [sample[2] for sample in skin_samples.values()]
    avg_a = np.mean(a_values)
    avg_b = np.mean(b_values)
    
    if abs(avg_a) > 15 or abs(avg_b) > 20:
        issues.append(("WARNING", "Strong color cast detected in image. Try neutral lighting."))
    
    # 3. Makeup detection (enhanced)
    forehead_samples = [v for k, v in skin_samples.items() if 'forehead' in k]
    cheek_samples = [v for k, v in skin_samples.items() if 'cheek' in k]
    
    if forehead_samples and cheek_samples:
        avg_forehead_a = np.mean([s[1] for s in forehead_samples])
        avg_cheek_a = np.mean([s[1] for s in cheek_samples])
        
        if avg_cheek_a > avg_forehead_a + 8:
            issues.append(("WARNING", "Heavy blush detected. Consider removing makeup for best accuracy."))
        elif avg_cheek_a > avg_forehead_a + 5:
            issues.append(("INFO", "Possible light makeup detected. May slightly affect accuracy."))
    
    # 4. Face size check (estimate based on skin sample positions)
    face_area_ratio = estimate_face_area_ratio(skin_samples, h, w)
    if face_area_ratio < 0.15:
        issues.append(("WARNING", "Face appears small in frame. Move closer to camera."))
    
    return issues

def estimate_face_area_ratio(skin_samples, h, w):
    """Estimate what proportion of image is face"""
    # This is a rough estimate - assumes ~9 samples across face
    # Each sample represents roughly 1% of face area
    estimated_face_ratio = len(skin_samples) * 0.02
    return estimated_face_ratio

def calculate_overall_confidence(temp_confidence, contrast_confidence, validation_issues):
    """Calculate confidence score for final result"""
    confidence_score = 100
    
    # Reduce based on feature confidence
    if temp_confidence == 'low':
        confidence_score -= 25
    elif temp_confidence == 'medium':
        confidence_score -= 10
    
    if contrast_confidence == 'low':
        confidence_score -= 15
    elif contrast_confidence == 'medium':
        confidence_score -= 5
    
    # Reduce based on validation issues
    for severity, _ in validation_issues:
        if severity == 'CRITICAL':
            confidence_score -= 30
        elif severity == 'WARNING':
            confidence_score -= 15
        elif severity == 'INFO':
            confidence_score -= 5
    
    confidence_level = 'High' if confidence_score >= 75 else \
                      'Medium' if confidence_score >= 50 else 'Low'
    
    recommendation = generate_confidence_message(confidence_score)
    
    return {
        'score': max(0, confidence_score),
        'level': confidence_level,
        'recommendation': recommendation
    }

def generate_confidence_message(score):
    """Generate user-friendly confidence message"""
    if score >= 85:
        return "Analysis confidence is very high. Results are highly reliable."
    elif score >= 75:
        return "Analysis confidence is high. Results are reliable."
    elif score >= 60:
        return "Analysis confidence is good. Results should be reasonably accurate."
    elif score >= 50:
        return "Analysis confidence is moderate. Consider retaking photo with better lighting."
    else:
        return "Analysis confidence is low. Please retake photo with better lighting and minimal makeup."

class ValidationMetrics:
    """Calculate accuracy metrics for testing"""
    
    def __init__(self):
        self.results = []
        self.confusion_matrix = {}
    
    def add_result(self, prediction, ground_truth, subject_id=None):
        """Add a single prediction result"""
        self.results.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'subject_id': subject_id,
            'correct_subseason': prediction['Sub_Season'] == ground_truth['subseason'],
            'correct_season': prediction['Season'] == ground_truth['season']
        })
    
    def calculate_accuracy(self):
        """Calculate overall accuracy metrics"""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        total = len(self.results)
        correct_subseason = sum(1 for r in self.results if r['correct_subseason'])
        correct_season = sum(1 for r in self.results if r['correct_season'])
        
        return {
            'subseason_accuracy': correct_subseason / total,
            'season_accuracy': correct_season / total,
            'total_samples': total,
            'correct_subseason': correct_subseason,
            'correct_season': correct_season
        }
    
    def analyze_failures(self):
        """Identify systematic errors"""
        failures = [r for r in self.results if not r['correct_subseason']]
        
        season_confusion = {}
        for failure in failures:
            pred = failure['prediction']['Season']
            actual = failure['ground_truth']['season']
            key = f"{actual}_predicted_as_{pred}"
            season_confusion[key] = season_confusion.get(key, 0) + 1
        
        return {
            'total_failures': len(failures),
            'season_confusion': season_confusion,
            'failure_rate': len(failures) / len(self.results) if self.results else 0,
            'detailed_failures': failures
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        accuracy = self.calculate_accuracy()
        failures = self.analyze_failures()
        
        report = f"""
=== COLOR ANALYSIS VALIDATION REPORT ===

Overall Accuracy:
- Sub-season accuracy: {accuracy['subseason_accuracy']:.2%} ({accuracy['correct_subseason']}/{accuracy['total_samples']})
- Season family accuracy: {accuracy['season_accuracy']:.2%} ({accuracy['correct_season']}/{accuracy['total_samples']})

Failure Analysis:
- Total failures: {failures['total_failures']}
- Failure rate: {failures['failure_rate']:.2%}

Season Confusion Patterns:
"""
        for pattern, count in failures['season_confusion'].items():
            report += f"  {pattern}: {count} occurrences\n"
        
        return report

class SystemTester:
    """Run systematic tests on the color analysis system"""
    
    def __init__(self, extractor, analyst):
        self.extractor = extractor
        self.analyst = analyst
        self.metrics = ValidationMetrics()
    
    def test_single_image(self, img_path, ground_truth, is_dyed=False):
        """Test a single image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        try:
            # Extract features
            features = self.extractor.extract_features_from_memory(img, is_dyed=is_dyed)
            
            # Analyze
            result = self.analyst.analyze_from_lab(
                features['skin_lab'],
                features['hair_lab'],
                features['eye_lab']
            )
            
            # Record result
            self.metrics.add_result(result, ground_truth, subject_id=img_path)
            
            return {
                'success': True,
                'prediction': result,
                'ground_truth': ground_truth
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image': img_path
            }
    
    def run_test_suite(self, test_dataset_path):
        """Run full validation suite from JSON test data"""
        with open(test_dataset_path, 'r') as f:
            test_data = json.load(f)
        
        results = []
        for subject in test_data['subjects']:
            for img_path in subject['images']:
                result = self.test_single_image(
                    img_path,
                    ground_truth={
                        'season': subject['season'],
                        'subseason': subject['known_season']
                    },
                    is_dyed=subject.get('dyed_hair', False)
                )
                results.append(result)
        
        return {
            'results': results,
            'report': self.metrics.generate_report(),
            'accuracy': self.metrics.calculate_accuracy(),
            'failures': self.metrics.analyze_failures()
        }

def create_test_dataset_template():
    """Create a template JSON for test dataset"""
    template = {
        "subjects": [
            {
                "id": "001",
                "name": "Example Subject A",
                "season": "Spring",
                "known_season": "Bright Spring",
                "images": [
                    "test_data/subject_001_front.jpg",
                    "test_data/subject_001_side.jpg"
                ],
                "professional_analysis": "Bright Spring - High contrast, warm undertone, clear eyes",
                "dyed_hair": False,
                "notes": "Verified by professional colorist"
            },
            {
                "id": "002",
                "name": "Example Subject B",
                "season": "Winter",
                "known_season": "True Winter",
                "images": [
                    "test_data/subject_002_front.jpg"
                ],
                "professional_analysis": "True Winter - Cool undertone, high contrast",
                "dyed_hair": False,
                "notes": "Self-identified, confirmed with palette test"
            }
        ],
        "metadata": {
            "created_date": "2026-01-14",
            "total_subjects": 2,
            "validation_method": "Professional color analysis",
            "notes": "Add more subjects for comprehensive testing"
        }
    }
    return template

if __name__ == "__main__":
    # Create test dataset template
    template = create_test_dataset_template()
    with open('/home/claude/test_dataset_template.json', 'w') as f:
        json.dump(template, f, indent=2)
    print("Created test_dataset_template.json")