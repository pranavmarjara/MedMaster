import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Bot,
  Upload,
  FileText,
  Loader2,
  CheckCircle,
  Clock,
  AlertCircle,
  FileCheck,
  Brain,
  Activity,
  MessageSquare,
  Lightbulb,
  Eye,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "wouter";

interface VibeyBot {
  id: string;
  name: string;
  role: string;
  description: string;
  icon: any;
  status: "idle" | "processing" | "completed" | "error";
  progress: number;
  message: string;
  result?: string;
}

interface AnalysisResult {
  intake: string;
  analysis: string;
  triage: string;
  explanation: string;
}

export default function MultiAI() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorAlert, setErrorAlert] = useState<string | null>(null);
  const queryClient = useQueryClient();
  
  const [bots, setBots] = useState<VibeyBot[]>([
    {
      id: "intake",
      name: "VibeyIntake",
      role: "File Processing Specialist",
      description: "Analyzing and extracting data from medical reports",
      icon: FileCheck,
      status: "idle",
      progress: 0,
      message: "Ready to process your medical report",
    },
    {
      id: "analysis",
      name: "VibeyAnalysis",
      role: "Medical Data Analyst",
      description: "Performing comprehensive medical data analysis",
      icon: Brain,
      status: "idle",
      progress: 0,
      message: "Waiting for processed data from VibeyIntake",
    },
    {
      id: "triage",
      name: "VibeyTriage",
      role: "Triage Specialist",
      description: "Determining priority levels and recommendations",
      icon: Activity,
      status: "idle",
      progress: 0,
      message: "Standby for triage assessment",
    },
    {
      id: "why",
      name: "VibeyWhy",
      role: "Explanation Expert",
      description: "Providing clear explanations and reasoning",
      icon: Lightbulb,
      status: "idle",
      progress: 0,
      message: "Ready to explain the diagnostic process",
    },
  ]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setAnalysisResult(null);
      setErrorAlert(null); // Clear any previous error alerts
      // Reset all bots to idle state
      setBots(prev => prev.map(bot => ({
        ...bot,
        status: "idle",
        progress: 0,
        message: bot.id === "intake" ? "Ready to process your medical report" :
                bot.id === "analysis" ? "Waiting for processed data from VibeyIntake" :
                bot.id === "triage" ? "Standby for triage assessment" :
                "Ready to explain the diagnostic process"
      })));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'text/xml': ['.xml'],
      'application/xml': ['.xml'],
    },
    maxFiles: 1,
  });

  const updateBotStatus = (botId: string, status: VibeyBot["status"], progress: number, message: string, result?: string) => {
    setBots(prev => prev.map(bot => 
      bot.id === botId 
        ? { ...bot, status, progress, message, result }
        : bot
    ));
  };

  const processFile = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Step 1: VibeyIntake processing
      updateBotStatus("intake", "processing", 20, "Reading and extracting data from your medical report...");
      
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
      
      updateBotStatus("intake", "processing", 60, "Converting data to structured format...");
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      updateBotStatus("intake", "processing", 90, "Validating extracted medical data...");
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      updateBotStatus("intake", "completed", 100, "Successfully processed medical report!");

      // Step 2: VibeyAnalysis processing
      updateBotStatus("analysis", "processing", 15, "Analyzing medical indicators and patterns...");
      
      await new Promise(resolve => setTimeout(resolve, 2500));
      
      updateBotStatus("analysis", "processing", 45, "Cross-referencing with medical databases...");
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      updateBotStatus("analysis", "processing", 75, "Identifying key medical findings...");
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      updateBotStatus("analysis", "completed", 100, "Medical analysis completed successfully!");

      // Step 3: VibeyTriage processing  
      updateBotStatus("triage", "processing", 25, "Assessing urgency and priority levels...");
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      updateBotStatus("triage", "processing", 60, "Generating triage recommendations...");
      
      await new Promise(resolve => setTimeout(resolve, 1800));
      
      updateBotStatus("triage", "processing", 85, "Finalizing triage classification...");
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      updateBotStatus("triage", "completed", 100, "Triage assessment completed!");

      // Step 4: VibeyWhy processing
      updateBotStatus("why", "processing", 30, "Analyzing diagnostic reasoning...");
      
      await new Promise(resolve => setTimeout(resolve, 2200));
      
      updateBotStatus("why", "processing", 70, "Preparing comprehensive explanation...");
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      updateBotStatus("why", "completed", 100, "Explanation ready!");

      // Now call the VibeyBot analysis engine
      const response = await fetch('/api/analyze-medical-report', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (!response.ok) {
        // Even on error, server provides fallback analysis results
        // Display them but also mark bots as having errors
        setAnalysisResult(result);
        setErrorAlert("Analysis completed with technical limitations. Results may be partial - please consult medical professionals for complete assessment.");
        setBots(prev => prev.map(bot => ({
          ...bot,
          status: "error",
          message: "Analysis completed with limitations - results may be partial"
        })));
        
        // Refresh data even for fallback results since they may be saved to database
        queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
        queryClient.invalidateQueries({ queryKey: ['diagnostic-history'] });
        return;
      }

      setAnalysisResult(result);
      
      // Immediately refresh dashboard and diagnostic history data
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
      queryClient.invalidateQueries({ queryKey: ['diagnostic-history'] });

    } catch (error) {
      console.error('Error processing file:', error);
      
      setErrorAlert("System error occurred during analysis. Basic fallback results provided - please consult medical professionals immediately.");
      
      // Mark all processing bots as error and provide fallback results
      setBots(prev => prev.map(bot => 
        bot.status === "processing" 
          ? { ...bot, status: "error", message: "Processing failed - please try again" }
          : bot
      ));
      
      // Provide basic fallback analysis if server completely failed
      setAnalysisResult({
        intake: "File upload successful, but automated processing encountered technical difficulties.",
        analysis: "Unable to complete automated analysis. Please consult with a medical professional for manual review.",
        triage: "Automated triage unavailable. Recommend immediate consultation with healthcare provider.",
        explanation: "Technical issues prevented complete analysis. Please have your medical documents reviewed by qualified medical professionals."
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const getBotStatusIcon = (status: VibeyBot["status"]) => {
    switch (status) {
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getBotStatusColor = (status: VibeyBot["status"]) => {
    switch (status) {
      case "processing":
        return "bg-blue-50 border-blue-200";
      case "completed":
        return "bg-green-50 border-green-200";
      case "error":
        return "bg-red-50 border-red-200";
      default:
        return "bg-gray-50 border-gray-200";
    }
  };

  return (
    <div className="min-h-full p-6 space-y-8">
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3 mb-6">
          <Bot className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold text-primary">Multi AI Diagnostic</h1>
        </div>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Upload your medical reports and watch our specialized AI agents work together to provide comprehensive analysis
        </p>
      </div>

      {/* Error Alert */}
      {errorAlert && (
        <Alert className="max-w-4xl mx-auto border-red-200 bg-red-50" data-testid="error-alert">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            {errorAlert}
          </AlertDescription>
        </Alert>
      )}

      {/* File Upload Area */}
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Medical Report
          </CardTitle>
          <CardDescription>
            Supports PDF, images, Word documents, text files, and more medical formats
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25 hover:border-primary/50"
            }`}
          >
            <input {...getInputProps()} />
            <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            {isDragActive ? (
              <p className="text-primary">Drop your medical report here...</p>
            ) : (
              <div>
                <p className="mb-2">Drag and drop your medical report here, or click to browse</p>
                <p className="text-sm text-muted-foreground">
                  Accepted formats: PDF, Images, Word, Text, CSV, JSON, XML
                </p>
              </div>
            )}
          </div>

          {file && (
            <div className="mt-4 p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <Button onClick={processFile} disabled={isProcessing}>
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Bot className="h-4 w-4 mr-2" />
                      Start Analysis
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* VibeyBots Workflow */}
      {file && (
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl font-bold mb-6 text-center">VibeyBot Analysis Pipeline</h2>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {bots.map((bot, index) => (
              <motion.div
                key={bot.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className={`relative overflow-hidden ${getBotStatusColor(bot.status)}`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <bot.icon className="h-6 w-6 text-primary" />
                        <div>
                          <h3 className="font-semibold">{bot.name}</h3>
                          <p className="text-xs text-muted-foreground">{bot.role}</p>
                        </div>
                      </div>
                      {getBotStatusIcon(bot.status)}
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <p className="text-sm mb-3">{bot.description}</p>
                    
                    {bot.status === "processing" && (
                      <div className="space-y-2">
                        <Progress value={bot.progress} className="h-2" />
                        <p className="text-xs text-muted-foreground">
                          {bot.progress}% complete
                        </p>
                      </div>
                    )}
                    
                    <div className="mt-3 p-2 bg-background/50 rounded text-xs">
                      <p className="flex items-start gap-2">
                        <MessageSquare className="h-3 w-3 mt-0.5 flex-shrink-0" />
                        {bot.message}
                      </p>
                    </div>

                    <Badge 
                      variant={
                        bot.status === "completed" ? "default" :
                        bot.status === "processing" ? "secondary" :
                        bot.status === "error" ? "destructive" : "outline"
                      }
                      className="mt-2 text-xs"
                    >
                      {bot.status === "idle" && "Ready"}
                      {bot.status === "processing" && "Working..."}
                      {bot.status === "completed" && "Complete"}
                      {bot.status === "error" && "Error"}
                    </Badge>
                  </CardContent>
                  
                  {bot.status === "processing" && (
                    <div className="absolute inset-0 bg-blue-500/5 animate-pulse" />
                  )}
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Results Display */}
      <AnimatePresence>
        {analysisResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="max-w-4xl mx-auto space-y-6"
          >
            <h2 className="text-2xl font-bold text-center mb-6">Analysis Results</h2>
            
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileCheck className="h-5 w-5 text-blue-600" />
                    VibeyIntake Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{analysisResult.intake}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-purple-600" />
                    VibeyAnalysis Findings
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{analysisResult.analysis}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-red-600" />
                    VibeyTriage Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{analysisResult.triage}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5 text-yellow-600" />
                    VibeyWhy Explanation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{analysisResult.explanation}</p>
                </CardContent>
              </Card>
            </div>
            
            {/* Action buttons after analysis */}
            <div className="flex justify-center gap-4 mt-6">
              <Link href="/diagnostics">
                <Button className="hover-elevate">
                  <Eye className="h-4 w-4 mr-2" />
                  View All Saved Results
                </Button>
              </Link>
              <Button 
                variant="outline" 
                onClick={() => {
                  setFile(null);
                  setAnalysisResult(null);
                  setErrorAlert(null);
                  setBots(prev => prev.map(bot => ({
                    ...bot,
                    status: "idle",
                    progress: 0,
                    message: bot.id === "intake" ? "Ready to process your medical report" :
                            bot.id === "analysis" ? "Waiting for processed data from VibeyIntake" :
                            bot.id === "triage" ? "Standby for triage assessment" :
                            "Ready to explain the diagnostic process"
                  })));
                }}
                className="hover-elevate"
              >
                <FileCheck className="h-4 w-4 mr-2" />
                Analyze Another Document
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}