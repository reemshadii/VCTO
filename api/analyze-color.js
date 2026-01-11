const fetch = require('node-fetch');

module.exports = async (req, res) => {
    // Only allow POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const { image, imageType, measurementMethod } = req.body;
        
        if (!image) {
            return res.status(400).json({ error: 'No image provided' });
        }

        console.log('Received color analysis request');
        console.log('Measurement method:', measurementMethod);

        // Build the prompt
        let analysisPrompt = `You are a professional color analyst. Analyze this person's coloring for seasonal color analysis. Look at their skin tone, hair color, and eye color to determine their season.

`;
        
        if (measurementMethod === 'ai') {
            analysisPrompt += `Also estimate their body measurements based on their visible proportions for virtual clothing try-on.

`;
        }
        
        analysisPrompt += `Provide ONLY a valid JSON response with no markdown code blocks, no explanation text before or after. Start directly with { and end with }.

CRITICAL: Analyze the ACTUAL person in the image. Do not default to any particular season. Consider:
- Skin undertone (warm yellow/peach vs cool pink/blue)
- Hair color warmth
- Eye color
- Overall contrast between features

IMPORTANT: The palette must contain 5 DIVERSE, SATURATED colors that are distinctly different from each other. Each color should be visually distinct and vibrant.

Color Requirements by Season:
- Spring: Warm, clear, bright colors (coral #ff7f50, peach #ffb347, golden yellow #ffd700, warm green #90ee90, turquoise #40e0d0)
- Summer: Cool, soft, muted colors (soft blue #6495ed, lavender #e6e6fa, rose pink #ffb6c1, mauve #d8bfd8, cool gray #708090)
- Autumn: Warm, deep, muted colors (rust #b7410e, olive green #6b8e23, mustard #ffdb58, burnt orange #cc5500, teal #008080)
- Winter: Cool, deep, clear colors (deep navy #000080, ruby red #e0115f, emerald green #50c878, royal purple #7851a9, magenta #ff00ff)

Response format:
{
  "undertone": "Warm" or "Cool" or "Neutral",
  "season": "Spring" or "Summer" or "Autumn" or "Winter",
  "subseason": "True" or "Bright" or "Soft" or "Dark",
  "contrast": "High" or "Medium" or "Low",
  "palette": ["#hexcolor1", "#hexcolor2", "#hexcolor3", "#hexcolor4", "#hexcolor5"]`;

        if (measurementMethod === 'ai') {
            analysisPrompt += `,
  "bodyMeasurements": {
    "bust": "number only, in cm",
    "waist": "number only, in cm",
    "hips": "number only, in cm",
    "shoulderWidth": "number only, in cm",
    "sleeveLength": "number only, in cm",
    "verticalLength": "number only, in cm"
  }`;
        }

        analysisPrompt += `
}`;

        // Call Anthropic API
        console.log('Calling Anthropic API...');
        
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
        console.log('API response received successfully');

        return res.status(200).json(data);

    } catch (error) {
        console.error('Function error:', error);
        return res.status(500).json({ 
            error: error.message || 'Internal server error',
            details: 'Check function logs for more information'
        });
    }
};