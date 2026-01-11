const fetch = require('node-fetch');

module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const { clothingImage, imageType, palette, season, undertone } = req.body;
        
        if (!clothingImage || !palette) {
            return res.status(400).json({ error: 'Missing required parameters' });
        }

        console.log('Analyzing clothing fit for', season, undertone);

        const prompt = `Analyze if this clothing item's colors fit within this palette: ${JSON.stringify(palette)}.
The person's season is ${season} (${undertone} undertone).

Provide ONLY a JSON response:
{
  "fits": true or false,
  "dominantColor": "#hexcolor",
  "reason": "brief explanation"
}

Return fits: true if the clothing's dominant colors match or complement the palette, false otherwise.`;

        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': process.env.ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify({
                model: 'claude-sonnet-4-20250514',
                max_tokens: 500,
                messages: [
                    {
                        role: 'user',
                        content: [
                            {
                                type: 'image',
                                source: {
                                    type: 'base64',
                                    media_type: imageType || 'image/jpeg',
                                    data: clothingImage
                                }
                            },
                            {
                                type: 'text',
                                text: prompt
                            }
                        ]
                    }
                ]
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();

        return res.status(200).json(data);

    } catch (error) {
        console.error('Function error:', error);
        return res.status(500).json({ 
            error: error.message || 'Internal server error'
        });
    }
};