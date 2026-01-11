const fetch = require('node-fetch');

// STRICT seasonal color logic - ONLY these 12 combinations exist
const COLOR_LOGIC = {
    "Deep Winter": {
        subseason: "Deep",
        season: "Winter",
        undertone: "Cool",
        contrast: "High",
        characteristics: "Dark, Cool & Clear",
        palette: ["#1D1D1B", "#FFFFFF", "#4A0E0E", "#002366", "#C0C0C0"],
        avoid: "Earth tones, golden oranges, and dusty pastels."
    },
    "True Winter": {
        subseason: "True",
        season: "Winter",
        undertone: "Cool",
        contrast: "High",
        characteristics: "Vivid, Chilly & Clear",
        palette: ["#000000", "#F5F5F5", "#D70270", "#0000FF", "#008080"],
        avoid: "Orange, Copper, Peach, and Mustard."
    },
    "Bright Winter": {
        subseason: "Bright",
        season: "Winter",
        undertone: "Cool",
        contrast: "High",
        characteristics: "Electric, Striking & Clear",
        palette: ["#000000", "#FFFFFF", "#FBEE0F", "#BC243C", "#00CED1"],
        avoid: "Muted, 'muddy' colors, and beige."
    },
    "Deep Autumn": {
        subseason: "Deep",
        season: "Autumn",
        undertone: "Warm",
        contrast: "Medium",
        characteristics: "Dark, Rich & Earthy",
        palette: ["#3D2B1F", "#F5F5DC", "#800000", "#556B2F", "#DAA520"],
        avoid: "Icy pastels, neon colors, and cool blues."
    },
    "True Autumn": {
        subseason: "True",
        season: "Autumn",
        undertone: "Warm",
        contrast: "Medium",
        characteristics: "Golden, Spiced & Rich",
        palette: ["#D2691E", "#E3A857", "#808000", "#A0522D", "#6B8E23"],
        avoid: "Black, Bright White, and Magenta."
    },
    "Soft Autumn": {
        subseason: "Soft",
        season: "Autumn",
        undertone: "Warm",
        contrast: "Low",
        characteristics: "Muted, Gentle & Dusty",
        palette: ["#8E7618", "#E6D5AC", "#BC8F8F", "#778899", "#8B864E"],
        avoid: "Vibrant neons, black, and high-contrast patterns."
    },
    "Bright Spring": {
        subseason: "Bright",
        season: "Spring",
        undertone: "Warm",
        contrast: "High",
        characteristics: "Lively, Vibrant & Warm",
        palette: ["#FF4500", "#FFF700", "#00FF7F", "#FF1493", "#00CED1"],
        avoid: "Dark, heavy colors and dusty, muted tones."
    },
    "True Spring": {
        subseason: "True",
        season: "Spring",
        undertone: "Warm",
        contrast: "Medium",
        characteristics: "Sunny, Warm & Clear",
        palette: ["#FFA500", "#FFD700", "#ADFF2F", "#F08080", "#20B2AA"],
        avoid: "Cool Greys, Navy, and Black."
    },
    "Light Spring": {
        subseason: "Light",
        season: "Spring",
        undertone: "Warm",
        contrast: "Low",
        characteristics: "Pale, Bright & Delicate",
        palette: ["#FFFACD", "#E0FFFF", "#FFB6C1", "#98FB98", "#FFA07A"],
        avoid: "Deep Burgundy, Black, and heavy Earth tones."
    },
    "Light Summer": {
        subseason: "Light",
        season: "Summer",
        undertone: "Cool",
        contrast: "Low",
        characteristics: "Pale, Cool & Delicate",
        palette: ["#F0F8FF", "#FFC0CB", "#B0E0E6", "#D8BFD8", "#AFEEEE"],
        avoid: "Heavy Dark colors, Neon Orange, and Gold."
    },
    "True Summer": {
        subseason: "True",
        season: "Summer",
        undertone: "Cool",
        contrast: "Low",
        characteristics: "Soft, Chilly & Muted",
        palette: ["#708090", "#957DAD", "#4A69BD", "#E090AD", "#576574"],
        avoid: "Yellow, Orange, Brown, and Warm Greens."
    },
    "Soft Summer": {
        subseason: "Soft",
        season: "Summer",
        undertone: "Cool",
        contrast: "Low",
        characteristics: "Muted, Smokey & Cool",
        palette: ["#696969", "#778899", "#BC8F8F", "#483D8B", "#2F4F4F"],
        avoid: "Electric colors, Neons, and Stark Black."
    }
};

module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const { image, imageType, measurementMethod } = req.body;
        
        if (!image) {
            return res.status(400).json({ error: 'No image provided' });
        }

        console.log('Received color analysis request');

        // Build the prompt with ABSOLUTE STRICTNESS
        let analysisPrompt = `You are a professional color analyst using the 12-season system.

═══════════════════════════════════════════════════════════════
CRITICAL: ONLY THESE 12 TYPES EXIST - NO EXCEPTIONS, NO MIXING
═══════════════════════════════════════════════════════════════

WINTER (All Cool undertone, All High contrast):
1. Deep Winter - Cool, High Contrast, Dark/Rich/Clear
2. True Winter - Cool, High Contrast, Vivid/Chilly/Clear  
3. Bright Winter - Cool, High Contrast, Electric/Striking

AUTUMN (All Warm undertone):
4. Deep Autumn - Warm, Medium Contrast, Dark/Rich/Earthy
5. True Autumn - Warm, Medium Contrast, Golden/Spiced/Rich
6. Soft Autumn - Warm, Low Contrast, Muted/Gentle/Dusty

SPRING (All Warm undertone):
7. Bright Spring - Warm, High Contrast, Lively/Vibrant
8. True Spring - Warm, Medium Contrast, Sunny/Clear
9. Light Spring - Warm, Low Contrast, Pale/Bright/Delicate

SUMMER (All Cool undertone, All Low contrast):
10. Light Summer - Cool, Low Contrast, Pale/Cool/Delicate
11. True Summer - Cool, Low Contrast, Soft/Chilly/Muted
12. Soft Summer - Cool, Low Contrast, Muted/Smokey/Cool

═══════════════════════════════════════════════════════════════
ABSOLUTE RULES - CANNOT BE BROKEN:
═══════════════════════════════════════════════════════════════

1. UNDERTONE RULES (NEVER violate these):
   ✓ Winter = ALWAYS Cool (NEVER Warm or Neutral)
   ✓ Summer = ALWAYS Cool (NEVER Warm or Neutral)  
   ✓ Spring = ALWAYS Warm (NEVER Cool or Neutral)
   ✓ Autumn = ALWAYS Warm (NEVER Cool or Neutral)

2. CONTRAST RULES (NEVER violate these):
   ✓ ALL Winter = High Contrast ONLY
   ✓ ALL Summer = Low Contrast ONLY
   ✓ Spring = High (Bright), Medium (True), or Low (Light)
   ✓ Autumn = Medium (Deep/True) or Low (Soft)

3. SUBSEASON RULES (NEVER violate these):
   ✓ Deep = ONLY Winter OR Autumn (never Spring/Summer)
   ✓ Bright = ONLY Winter OR Spring (never Summer/Autumn)
   ✓ Light = ONLY Spring OR Summer (never Winter/Autumn)
   ✓ Soft = ONLY Autumn OR Summer (never Winter/Spring)
   ✓ True = Any season is valid

IMPOSSIBLE COMBINATIONS (DO NOT CREATE THESE):
❌ Warm Summer (Summer is ALWAYS Cool)
❌ Cool Spring (Spring is ALWAYS Warm)
❌ Cool Autumn (Autumn is ALWAYS Warm)
❌ Warm Winter (Winter is ALWAYS Cool)
❌ Bright Summer (Summer can only be Light/True/Soft)
❌ Bright Autumn (Autumn can only be Deep/True/Soft)
❌ Deep Spring (Spring can only be Bright/True/Light)
❌ Deep Summer (Summer can only be Light/True/Soft)
❌ Light Winter (Winter can only be Deep/True/Bright)
❌ Light Autumn (Autumn can only be Deep/True/Soft)
❌ Soft Winter (Winter can only be Deep/True/Bright)
❌ Soft Spring (Spring can only be Bright/True/Light)

═══════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════

Analyze this person's:
- Skin undertone (pink/blue = Cool, yellow/peach = Warm)
- Hair color warmth
- Eye color
- Contrast between features

Pick EXACTLY ONE of the 12 valid types above.

Return ONLY this JSON (no markdown, no explanation):
{
  "undertone": "Warm" or "Cool",
  "season": "Winter", "Summer", "Autumn", or "Spring",
  "subseason": "Deep", "True", "Bright", "Soft", or "Light",
  "contrast": "High", "Medium", or "Low",
  "palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"]`;

        if (measurementMethod === 'ai') {
            analysisPrompt += `,
  "bodyMeasurements": {
    "bust": "number in cm",
    "waist": "number in cm",
    "hips": "number in cm",
    "shoulderWidth": "number in cm",
    "sleeveLength": "number in cm",
    "verticalLength": "number in cm"
  }`;
        }

        analysisPrompt += `
}

Double-check your answer matches ONE of the 12 valid types before responding.`;

        // Call Anthropic API
        console.log('Calling Anthropic API with strict validation...');
        
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': process.env.ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify({
                model: 'claude-sonnet-4-20250514',
                max_tokens: 1500,
                messages: [
                    {
                        role: 'user',
                        content: [
                            {
                                type: 'image',
                                source: {
                                    type: 'base64',
                                    media_type: imageType || 'image/jpeg',
                                    data: image
                                }
                            },
                            {
                                type: 'text',
                                text: analysisPrompt
                            }
                        ]
                    }
                ]
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Anthropic API error:', response.status, errorText);
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();
        
        // Extract and parse response
        let responseText = "";
        if (data.content && Array.isArray(data.content)) {
            responseText = data.content
                .filter(item => item.type === "text")
                .map(item => item.text)
                .join("");
        }

        const cleanJson = responseText.replace(/```json\n?|```/g, '').trim();
        const apiResults = JSON.parse(cleanJson);
        
        // ═══════════════════════════════════════════════════════════════
        // VALIDATION: Force correct values if AI made a mistake
        // ═══════════════════════════════════════════════════════════════
        
        const seasonKey = `${apiResults.subseason} ${apiResults.season}`;
        console.log('AI suggested:', seasonKey);
        
        if (!COLOR_LOGIC[seasonKey]) {
            console.error('❌ INVALID COMBINATION:', seasonKey);
            console.log('Forcing to valid default...');
            
            // Force to valid default based on what we can salvage
            if (apiResults.season === 'Winter' || apiResults.season === 'Summer') {
                apiResults.undertone = 'Cool'; // Winter/Summer are ALWAYS Cool
            } else {
                apiResults.undertone = 'Warm'; // Spring/Autumn are ALWAYS Warm
            }
            
            // Find closest valid season
            if (apiResults.undertone === 'Warm') {
                if (apiResults.contrast === 'High') {
                    apiResults.season = 'Spring';
                    apiResults.subseason = 'Bright';
                } else if (apiResults.contrast === 'Low') {
                    apiResults.season = 'Autumn';
                    apiResults.subseason = 'Soft';
                } else {
                    apiResults.season = 'Autumn';
                    apiResults.subseason = 'True';
                }
            } else {
                if (apiResults.contrast === 'High') {
                    apiResults.season = 'Winter';
                    apiResults.subseason = 'True';
                } else {
                    apiResults.season = 'Summer';
                    apiResults.subseason = 'True';
                    apiResults.contrast = 'Low'; // Summer is ALWAYS Low contrast
                }
            }
        } else {
            // Validate and correct the combination
            const validSeason = COLOR_LOGIC[seasonKey];
            apiResults.undertone = validSeason.undertone;
            apiResults.contrast = validSeason.contrast;
        }
        
        // Use the correct palette from our definitions
        const finalSeasonKey = `${apiResults.subseason} ${apiResults.season}`;
        if (COLOR_LOGIC[finalSeasonKey]) {
            apiResults.palette = COLOR_LOGIC[finalSeasonKey].palette;
            console.log('✓ Valid season:', finalSeasonKey);
        } else {
            // This should never happen, but just in case
            console.error('Still invalid after correction, using True Autumn as fallback');
            apiResults.undertone = 'Warm';
            apiResults.season = 'Autumn';
            apiResults.subseason = 'True';
            apiResults.contrast = 'Medium';
            apiResults.palette = COLOR_LOGIC['True Autumn'].palette;
        }

        console.log('Final result:', `${apiResults.subseason} ${apiResults.season} (${apiResults.undertone}, ${apiResults.contrast} Contrast)`);
        
        // Return in Claude API format
        return res.status(200).json({
            content: [
                {
                    type: "text",
                    text: JSON.stringify(apiResults)
                }
            ]
        });

    } catch (error) {
        console.error('Function error:', error);
        return res.status(500).json({ 
            error: error.message || 'Internal server error',
            details: 'Check function logs for more information'
        });
    }
};
