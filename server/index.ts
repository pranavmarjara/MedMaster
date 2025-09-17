import express, { type Request, Response, NextFunction } from "express";
import helmet from "helmet";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

const app = express();

// Security headers with environment-based CSP
const isDevelopment = app.get("env") === "development";

const cspDirectives = {
  defaultSrc: ["'self'"],
  scriptSrc: isDevelopment 
    ? ["'self'", "'unsafe-inline'", "'unsafe-eval'"] 
    : ["'self'"],
  styleSrc: isDevelopment 
    ? ["'self'", "'unsafe-inline'"] 
    : ["'self'"],
  imgSrc: ["'self'", "data:", "blob:"],
  connectSrc: isDevelopment 
    ? ["'self'", "ws:", "wss:"] 
    : ["'self'"],
  fontSrc: ["'self'"],
  objectSrc: ["'none'"],
  mediaSrc: ["'self'"],
  frameSrc: ["'none'"]
};

app.use(helmet({
  contentSecurityPolicy: {
    directives: cspDirectives
  },
  crossOriginEmbedderPolicy: !isDevelopment,
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Sanitize sensitive fields from logs
const sanitizeLogData = (data: any): any => {
  if (!data || typeof data !== 'object') return data;
  
  const sensitiveFields = ['password', 'token', 'secret', 'key', 'auth', 'ssn', 'dob', 'medical'];
  const sanitized = { ...data };
  
  for (const field in sanitized) {
    if (sensitiveFields.some(sensitive => field.toLowerCase().includes(sensitive))) {
      sanitized[field] = '[REDACTED]';
    } else if (typeof sanitized[field] === 'object' && sanitized[field] !== null) {
      sanitized[field] = sanitizeLogData(sanitized[field]);
    }
  }
  
  return sanitized;
};

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      
      // Only log sanitized response data if status indicates success and not in production
      if (capturedJsonResponse && res.statusCode < 400 && isDevelopment) {
        const sanitizedResponse = sanitizeLogData(capturedJsonResponse);
        logLine += ` :: ${JSON.stringify(sanitizedResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    
    // Log error securely (no sensitive data)
    const errorLog = {
      timestamp: new Date().toISOString(),
      method: req.method,
      path: req.path,
      status,
      message: err.message,
      stack: isDevelopment ? err.stack : undefined
    };
    console.error('Server Error:', JSON.stringify(errorLog));
    
    // Return generic error messages in production to avoid information disclosure
    const message = isDevelopment 
      ? err.message || "Internal Server Error"
      : status < 500 
        ? err.message || "Bad Request"
        : "Internal Server Error";

    res.status(status).json({ message });
    // Don't rethrow - this prevents server crashes
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || '5000', 10);
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true,
  }, () => {
    log(`serving on port ${port}`);
  });
})();
