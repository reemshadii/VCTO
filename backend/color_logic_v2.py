import numpy as np

class PersonalColorAnalystV2:
    
    def calculate_temperature_score_v2(self, skin_lab, hair_lab, eye_lab):
        temp_score = 0
        confidence_factors = []
        
        if skin_lab[2] > 18:
            score = 4
            temp_score += score
            confidence_factors.append(('skin_very_warm', score, 'high'))
        elif skin_lab[2] > 14:
            score = 3
            temp_score += score
            confidence_factors.append(('skin_warm', score, 'medium'))
        elif skin_lab[2] < 6:
            score = -4
            temp_score += score
            confidence_factors.append(('skin_cool', score, 'high'))
        elif skin_lab[2] < 10:
            score = -3
            temp_score += score
            confidence_factors.append(('skin_slightly_cool', score, 'medium'))
        else:
            confidence_factors.append(('skin_neutral', 0, 'low'))
        
        if skin_lab[1] < 2 and skin_lab[2] > 8:
            temp_score -= 2
            confidence_factors.append(('olive_modifier', -2, 'medium'))
        elif skin_lab[1] > 8:
            temp_score -= 1
            confidence_factors.append(('pink_modifier', -1, 'low'))
        
        if hair_lab[2] > 14:
            score = 2
            temp_score += score
            confidence_factors.append(('hair_warm', score, 'medium'))
        elif hair_lab[2] < 5:
            score = -2
            temp_score += score
            confidence_factors.append(('hair_cool', score, 'medium'))
        
        if eye_lab[2] > 10:
            score = 1
            temp_score += score
            confidence_factors.append(('eye_warm', score, 'low'))
        elif eye_lab[2] < 0:
            score = -1
            temp_score += score
            confidence_factors.append(('eye_cool', score, 'low'))
        
        high_conf_count = sum(1 for _, _, conf in confidence_factors if conf == 'high')
        confidence = 'high' if high_conf_count >= 2 else 'medium' if high_conf_count >= 1 else 'low'
        
        if temp_score > 3:
            temp_cat = "Warm"
        elif temp_score < -2:
            temp_cat = "Cool"
        else:
            temp_cat = "Neutral"
        
        return {
            'score': temp_score,
            'category': temp_cat,
            'confidence': confidence,
            'factors': confidence_factors
        }
    
    def calculate_contrast_v2(self, skin_lab, hair_lab, eye_lab):
        l_contrast = abs(skin_lab[0] - hair_lab[0]) + abs(skin_lab[0] - eye_lab[0])
        
        skin_chroma = np.sqrt(skin_lab[1]**2 + skin_lab[2]**2)
        hair_chroma = np.sqrt(hair_lab[1]**2 + hair_lab[2]**2)
        eye_chroma = np.sqrt(eye_lab[1]**2 + eye_lab[2]**2)
        
        chroma_contrast = abs(skin_chroma - hair_chroma) + abs(skin_chroma - eye_chroma)
        total_contrast = (l_contrast * 0.75) + (chroma_contrast * 0.25)
        
        if total_contrast > 70:
            category = "High"
            confidence = "high"
        elif total_contrast > 55:
            category = "Medium-High"
            confidence = "medium"
        elif total_contrast > 40:
            category = "Medium"
            confidence = "medium"
        elif total_contrast > 28:
            category = "Medium-Low"
            confidence = "medium"
        else:
            category = "Low"
            confidence = "high"
        
        return {
            'category': category,
            'value': total_contrast,
            'l_contrast': l_contrast,
            'chroma_contrast': chroma_contrast,
            'confidence': confidence
        }
    
    def determine_season_logic_v2(self, temp_data, contrast_data, skin_lab, hair_lab, eye_lab):
        temp_score = temp_data['score']
        temp_confidence = temp_data['confidence']
        contrast_val = contrast_data['value']
        contrast_cat = contrast_data['category']
        hair_l = hair_lab[0]
        
        season_scores = {'Winter': 0, 'Summer': 0, 'Spring': 0, 'Autumn': 0}
        
        if temp_score < -3:
            season_scores['Winter'] += 3
            season_scores['Summer'] += 2
        elif temp_score < 0:
            season_scores['Summer'] += 3
            season_scores['Winter'] += 1
        elif temp_score > 3:
            season_scores['Autumn'] += 3
            season_scores['Spring'] += 2
        else:
            season_scores['Spring'] += 3
            season_scores['Autumn'] += 1
        
        if hair_l < 25:
            season_scores['Winter'] += 2
            season_scores['Autumn'] += 2
        elif hair_l < 30:
            season_scores['Winter'] += 1
            season_scores['Autumn'] += 1
            if temp_score <= 1:
                season_scores['Summer'] += 2
        elif hair_l > 65:
            season_scores['Summer'] += 2
            season_scores['Spring'] += 2
        elif hair_l > 50:
            season_scores['Summer'] += 1
            season_scores['Spring'] += 1
        
        if contrast_val > 65:
            season_scores['Winter'] += 2
            season_scores['Spring'] += 1
        elif contrast_val < 35:
            season_scores['Summer'] += 2
            season_scores['Autumn'] += 1
        
        season = max(season_scores, key=season_scores.get)
        max_score = season_scores[season]
        second_best = sorted(season_scores.values(), reverse=True)[1]
        score_gap = max_score - second_best
        
        if score_gap >= 3:
            overall_confidence = "high"
        elif score_gap >= 2:
            overall_confidence = "medium"
        else:
            overall_confidence = "low"
        
        if temp_confidence == 'low' or contrast_data['confidence'] == 'low':
            if overall_confidence == 'high':
                overall_confidence = 'medium'
            elif overall_confidence == 'medium':
                overall_confidence = 'low'
        
        if season == "Winter":
            if temp_score > -2:
                subseason = "Deep Winter"
            elif contrast_val > 70:
                subseason = "Bright Winter"
            else:
                subseason = "True Winter"
        
        elif season == "Summer":
            if hair_l < 30:
                if contrast_val >= 35:
                    subseason = "True Summer"
                else:
                    subseason = "Soft Summer"
            elif hair_l > 60:
                subseason = "Light Summer"
            elif contrast_val < 40:
                subseason = "Soft Summer"
            else:
                subseason = "True Summer"
        
        elif season == "Spring":
            if contrast_val > 65:
                subseason = "Bright Spring"
            elif hair_l > 65:
                subseason = "Light Spring"
            else:
                subseason = "True Spring"
        
        else:
            if temp_score < 2:
                subseason = "Deep Autumn"
            elif contrast_val < 40:
                subseason = "Soft Autumn"
            else:
                subseason = "True Autumn"
        
        return {
            'season': season,
            'subseason': subseason,
            'confidence': overall_confidence,
            'season_scores': season_scores,
            'score_gap': score_gap
        }
    
    def analyze_from_lab(self, skin_lab, hair_lab, eye_lab):
        temp_data = self.calculate_temperature_score_v2(skin_lab, hair_lab, eye_lab)
        contrast_data = self.calculate_contrast_v2(skin_lab, hair_lab, eye_lab)
        season_data = self.determine_season_logic_v2(temp_data, contrast_data, skin_lab, hair_lab, eye_lab)
        
        return {
            'Season': season_data['season'],
            'Sub_Season': season_data['subseason'],
            'Temperature_Category': temp_data['category'],
            'Temperature_Score': temp_data['score'],
            'Contrast_Category': contrast_data['category'],
            'Contrast_Value': contrast_data['value'],
            'Confidence': season_data['confidence'],
            'Details': {
                'temperature': temp_data,
                'contrast': contrast_data,
                'season_scores': season_data['season_scores']
            }
        }
