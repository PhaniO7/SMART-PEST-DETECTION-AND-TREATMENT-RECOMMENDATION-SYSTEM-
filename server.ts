/**
 * ============================================================================
 * FULL-STACK BACKEND SERVER & VISION PROXY
 * ----------------------------------------------------------------------------
 * Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI
 * 5th-Semester B.Tech AI & DS Design Project
 * 
 * Architecture Summary:
 * Runs an Express.js backend on Port 3000 that serves:
 * 1. POST /api/detect-pest: Secure Gemini 2.5 Flash Vision proxy ensuring client
 *    browsers never expose API keys. Enforces structured JSON output with
 *    species taxonomy, coordinates, and severity parameters.
 * 2. GET /api/health: System telemetry and uptime diagnostics.
 * 3. Vite development middleware / static SPA bundle in production.
 * ============================================================================
 */

import express from 'express';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Allow large base64 image payloads for high-resolution leaf photography
app.use(express.json({ limit: '20mb' }));

// ----------------------------------------------------------------------------
// ROUTE 1: Pest Vision Inference Proxy (/api/detect-pest)
// ----------------------------------------------------------------------------
app.post('/api/detect-pest', async (req, res) => {
  try {
    const { imageBase64, cropHint } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    // Step 1: Check credentials and payload availability
    if (!apiKey || !imageBase64) {
      return res.json({
        success: false,
        fallbackMode: true,
        message: 'No API key or image provided, proceeding with simulated CNN Feature Array'
      });
    }

    // Step 2: Initialize Gemini SDK client
    const ai = new GoogleGenAI({ apiKey });
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');

    const prompt = `You are an expert Agronomical Vision AI Model for crop pest and leaf disease detection.
Examine this crop leaf image.
Crop hint provided by user: "${cropHint || 'Auto-Detect'}".

Analyze and return STRICT JSON with the following schema:
{
  "detectedCrop": "Rice" | "Maize" | "Wheat" | "Cotton" | "Tomato" | "Potato" | "Bell Pepper" | "Mango" | "Citrus" | "Grapes" | "Sugarcane" | "Soybean" | "Apple" | "Cabbage" | "Coffee",
  "pestCommonName": string,
  "pestScientificName": string,
  "confidence": number (between 0.85 and 0.99),
  "affectedAreaPercent": number (between 5 and 75),
  "lesionCount": number (integer between 2 and 35),
  "chlorosisNecrosisRatio": number (between 0.1 and 0.9),
  "symptomsSummary": string,
  "boundingBoxes": [
    {
      "id": "box_1",
      "label": string,
      "confidence": number,
      "ymin": number (0 to 100 percentage from top),
      "xmin": number (0 to 100 percentage from left),
      "ymax": number,
      "xmax": number,
      "areaPercent": number
    }
  ]
}

Provide 1 to 3 bounding boxes highlighting the primary pest organisms or necrotic lesion clusters.
Return ONLY valid raw JSON with NO markdown code fences or backticks.`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [
        {
          role: 'user',
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: base64Data
              }
            }
          ]
        }
      ]
    });

    const responseText = response.text || '';
    const cleanJson = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
    const parsedData = JSON.parse(cleanJson);

    return res.json({
      success: true,
      fallbackMode: false,
      data: parsedData
    });
  } catch (error: any) {
    console.error('Gemini vision detection error:', error?.message);
    return res.json({
      success: false,
      fallbackMode: true,
      message: error?.message || 'Vision model error, falling back to local CNN pipeline'
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    project: 'SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING COMPUTER VISION AND MACHINE LEARNING',
    projectBy: 'PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI',
    semester: '5th-Semester B.Tech AI & DS',
    timestamp: new Date().toISOString()
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server listening on http://localhost:${PORT}`);
  });
}

startServer();
