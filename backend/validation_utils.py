import cv2
import numpy as np

def detect_environmental_issues_v2(skin_samples, image_rgb, skin_lab):
    issues = []
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
    
    a_values = [sample[1] for sample in skin_samples.values()]
    b_values = [sample[2] for sample in skin_samples.values()]
    avg_a = np.mean(a_values)
    avg_b = np.mean(b_values)
    
    if abs(avg_a) > 15 or abs(avg_b) > 20:
        issues.append(("WARNING", "Strong color cast detected in image. Try neutral lighting."))
    
    forehead_samples = [v for k, v in skin_samples.items() if 'forehead' in k]
    cheek_samples = [v for k, v in skin_samples.items() if 'cheek' in k]
    
    if forehead_samples and cheek_samples:
        avg_forehead_a = np.mean([s[1] for s in forehead_samples])
        avg_cheek_a = np.mean([s[1] for s in cheek_samples])
        
        if avg_cheek_a > avg_forehead_a + 8:
            issues.append(("WARNING", "Heavy blush detected. Consider removing makeup for best accuracy."))
        elif avg_cheek_a > avg_forehead_a + 5:
            issues.append(("INFO", "Possible light makeup detected. May slightly affect accuracy."))
    
    return issues

def calculate_overall_confidence(temp_confidence, contrast_confidence, validation_issues):
    confidence_score = 100
    
    if temp_confidence == 'low':
        confidence_score -= 25
    elif temp_confidence == 'medium':
        confidence_score -= 10
    
    if contrast_confidence == 'low':
        confidence_score -= 15
    elif contrast_confidence == 'medium':
        confidence_score -= 5
    
    for severity, _ in validation_issues:
        if severity == 'CRITICAL':
            confidence_score -= 30
        elif severity == 'WARNING':
            confidence_score -= 15
        elif severity == 'INFO':
            confidence_score -= 5
    
    confidence_level = 'High' if confidence_score >= 75 else 'Medium' if confidence_score >= 50 else 'Low'
    
    if confidence_score >= 85:
        recommendation = "Analysis confidence is very high. Results are highly reliable."
    elif confidence_score >= 75:
        recommendation = "Analysis confidence is high. Results are reliable."
    elif confidence_score >= 60:
        recommendation = "Analysis confidence is good. Results should be reasonably accurate."
    elif confidence_score >= 50:
        recommendation = "Analysis confidence is moderate. Consider retaking photo with better lighting."
    else:
        recommendation = "Analysis confidence is low. Please retake photo with better lighting and minimal makeup."
    
    return {
        'score': max(0, confidence_score),
        'level': confidence_level,
        'recommendation': recommendation
    }
