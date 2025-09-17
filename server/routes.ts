import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { analyzeMedicalReport, analyzeImageReport } from "./gemini";

// Dynamic import for pdf-parse to avoid startup issues
async function parsePdf(buffer: Buffer): Promise<string> {
  try {
    const pdfParse = (await import("pdf-parse")).default;
    const data = await pdfParse(buffer);
    return data.text;
  } catch (error) {
    console.error('PDF parsing error:', error);
    throw error;
  }
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
            fileContent = await parsePdf(dataBuffer);
            console.log(`Extracted ${fileContent.length} characters from PDF: ${originalName}`);
          } catch (pdfError) {
            console.error('Error extracting PDF content:', pdfError);
            fileContent = `Medical PDF document: ${originalName} - text extraction failed. Please ensure the PDF contains extractable text.`;
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

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  const httpServer = createServer(app);

  return httpServer;
}
