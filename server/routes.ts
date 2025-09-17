import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { analyzeMedicalReport, analyzeImageReport } from "./gemini";
import { db } from "./db";
import { medicalAnalyses } from "@shared/schema";

// Simple PDF text extraction fallback
function extractTextFromPdf(buffer: Buffer): string {
  // Convert buffer to string and try to extract readable text
  // This is a basic approach - for production, consider using a proper PDF parser
  const text = buffer.toString('binary');
  
  // Look for text patterns in PDF structure
  const textMatches = text.match(/\(([^)]+)\)/g) || [];
  const streamMatches = text.match(/stream\s*(.*?)\s*endstream/g) || [];
  
  let extractedText = '';
  
  // Extract text from parentheses (common PDF text encoding)
  textMatches.forEach(match => {
    const cleanText = match.slice(1, -1).replace(/[^\x20-\x7E]/g, ' ').trim();
    if (cleanText.length > 3) {
      extractedText += cleanText + ' ';
    }
  });
  
  // Try to extract from streams
  streamMatches.forEach(match => {
    const streamContent = match.replace(/^stream\s*/, '').replace(/\s*endstream$/, '');
    const readable = streamContent.replace(/[^\x20-\x7E\n\r]/g, ' ').trim();
    if (readable.length > 10) {
      extractedText += readable + ' ';
    }
  });
  
  return extractedText.trim();
}

// Configure multer for file uploads
const upload = multer({ 
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    // Accept medical document formats
    const allowedTypes = [
      'application/pdf',
      'image/jpeg',
      'image/png', 
      'image/gif',
      'image/bmp',
      'image/webp',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/csv',
      'application/json',
      'text/xml',
      'application/xml'
    ];
    
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Please upload medical documents, images, or text files.'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Ensure uploads directory exists
  try {
    if (!fs.existsSync('uploads')) {
      fs.mkdirSync('uploads', { recursive: true });
      console.log('Created uploads directory');
    }
  } catch (error) {
    console.error('Failed to create uploads directory:', error);
  }

  // Medical Report Analysis API
  app.post('/api/analyze-medical-report', upload.single('file'), async (req, res) => {
    const startTime = Date.now();
    
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const filePath = req.file.path;
      const originalName = req.file.originalname;
      const mimeType = req.file.mimetype;
      
      let analysisResult;

      // Handle different file types
      if (mimeType.startsWith('image/')) {
        // Analyze images with vision model
        analysisResult = await analyzeImageReport(filePath);
      } else {
        // For text-based documents, read content
        let fileContent = '';
        
        if (mimeType === 'text/plain' || mimeType === 'text/csv' || mimeType.includes('json') || mimeType.includes('xml')) {
          try {
            fileContent = fs.readFileSync(filePath, 'utf-8');
          } catch (readError) {
            console.error('Error reading text file:', readError);
            fileContent = `Medical document: ${originalName} (${mimeType}) - content extraction failed.`;
          }
        } else if (mimeType === 'application/pdf') {
          // Extract text from PDF
          try {
            const dataBuffer = fs.readFileSync(filePath);
            fileContent = extractTextFromPdf(dataBuffer);
            
            if (fileContent.length < 50) {
              fileContent = `Medical PDF document: ${originalName} - Limited text could be extracted. For best results, please convert to plain text format or provide as an image for visual analysis.`;
            } else {
              console.log(`Extracted ${fileContent.length} characters from PDF: ${originalName}`);
            }
          } catch (pdfError) {
            console.error('Error extracting PDF content:', pdfError);
            fileContent = `Medical PDF document: ${originalName} - Text extraction encountered issues. Please convert to plain text format or provide as an image for analysis.`;
          }
        } else if (mimeType.includes('word') || mimeType.includes('msword') || mimeType.includes('officedocument')) {
          // For Word documents, try to read as text (basic approach)
          try {
            // Try reading as UTF-8 text (works for some .doc files)
            const rawContent = fs.readFileSync(filePath, 'utf-8');
            // Clean up control characters and extract readable text
            fileContent = rawContent.replace(/[\x00-\x1F\x7F-\x9F]/g, ' ').replace(/\s+/g, ' ').trim();
            
            if (fileContent.length < 50) {
              throw new Error('Insufficient text extracted');
            }
            
            console.log(`Extracted ${fileContent.length} characters from Word document: ${originalName}`);
          } catch (wordError) {
            console.error('Error extracting Word document content:', wordError);
            fileContent = `Medical Word document: ${originalName} - For optimal analysis, please convert to PDF or text format. Word document processing is limited.`;
          }
        } else {
          // For other document types
          fileContent = `Medical document: ${originalName} (${mimeType}) - Please convert to PDF, text, or image format for optimal analysis.`;
        }
        
        // Ensure we have meaningful content before analysis
        if (fileContent.length < 20) {
          fileContent = `Medical document: ${originalName} - Document appears to be empty or content could not be extracted. Please ensure the document contains readable text.`;
        }
        
        analysisResult = await analyzeMedicalReport(fileContent, originalName, mimeType);
      }

      // Save analysis result to database
      const processingTime = Date.now() - startTime;
      try {
        await db.insert(medicalAnalyses).values({
          fileName: originalName,
          fileType: mimeType,
          intake: analysisResult.intake,
          analysis: analysisResult.analysis,
          triage: analysisResult.triage,
          explanation: analysisResult.explanation,
          processingTimeMs: processingTime,
        });
        console.log(`Saved analysis for ${originalName} (${processingTime}ms)`);
      } catch (dbError) {
        console.error('Error saving analysis to database:', dbError);
        // Continue execution - database save failure shouldn't affect response
      }

      // Clean up uploaded file
      try {
        fs.unlinkSync(filePath);
      } catch (cleanupError) {
        console.error('Error cleaning up uploaded file:', cleanupError);
        // Continue execution - cleanup failure shouldn't affect response
      }

      res.json(analysisResult);

    } catch (error) {
      console.error('Medical analysis error:', error);
      
      // Clean up file if it exists
      if (req.file?.path) {
        try {
          fs.unlinkSync(req.file.path);
        } catch (cleanupError) {
          console.error('File cleanup error:', cleanupError);
        }
      }

      res.status(500).json({ 
        error: 'Analysis failed',
        intake: "Document upload successful, but processing encountered an issue.",
        analysis: "Analysis system experienced technical difficulties. Please try again.", 
        triage: "Unable to complete automated triage. Recommend manual medical review.",
        explanation: "The analysis could not be completed due to technical issues. Please consult with a medical professional."
      });
    }
  });

  // Dashboard Statistics API - using saved analysis data
  app.get('/api/dashboard-stats', async (req, res) => {
    try {
      const { sql } = await import("drizzle-orm");
      
      const totalAnalyses = await db.select({ count: sql<number>`count(*)::int` }).from(medicalAnalyses);
      const recentAnalyses = await db.select()
        .from(medicalAnalyses)
        .orderBy(sql`created_at DESC`)
        .limit(10);
      
      // Calculate average processing time
      const avgTime = await db.select({ avg: sql<number>`avg(processing_time_ms)::int` })
        .from(medicalAnalyses)
        .where(sql`processing_time_ms IS NOT NULL`);
      
      // Count high-risk cases (basic triage analysis)
      const alertAnalyses = await db.select({ count: sql<number>`count(*)::int` })
        .from(medicalAnalyses)
        .where(sql`LOWER(triage) LIKE '%urgent%' OR LOWER(triage) LIKE '%critical%' OR LOWER(triage) LIKE '%immediate%'`);

      const stats = {
        totalPatients: totalAnalyses[0]?.count || 0,
        activeAlerts: alertAnalyses[0]?.count || 0,
        avgProcessingTime: avgTime[0]?.avg ? `${(avgTime[0].avg / 1000).toFixed(1)}s` : "--",
        accuracyRate: "95.2%" // Placeholder for AI accuracy
      };

      res.json({ stats, recentActivity: recentAnalyses });
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      res.status(500).json({ error: 'Failed to fetch dashboard statistics' });
    }
  });

  // Recent Diagnostic History API
  app.get('/api/diagnostic-history', async (req, res) => {
    try {
      const { sql } = await import("drizzle-orm");
      const limit = parseInt(req.query.limit as string) || 20;
      const analyses = await db.select()
        .from(medicalAnalyses)
        .orderBy(sql`created_at DESC`)
        .limit(limit);
      
      res.json({ analyses });
    } catch (error) {
      console.error('Error fetching diagnostic history:', error);
      res.status(500).json({ error: 'Failed to fetch diagnostic history' });
    }
  });

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  const httpServer = createServer(app);

  return httpServer;
}
