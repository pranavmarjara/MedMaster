import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Brain, TrendingUp, Eye, Clock, CheckCircle, FileText, Calendar } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

interface Diagnosis {
  condition: string;
  confidence: number;
  evidence: string[];
  severity: "low" | "medium" | "high";
  icd10: string;
}

// Fetch diagnostic history from API
const fetchDiagnosticHistory = async () => {
  const response = await fetch('/api/diagnostic-history?limit=10');
  if (!response.ok) {
    throw new Error('Failed to fetch diagnostic history');
  }
  return response.json();
};

// Real diagnosis data will be populated from AI analysis
const emptyDiagnoses: Diagnosis[] = [];

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "high": return "text-destructive";
    case "medium": return "text-chart-2";
    case "low": return "text-chart-3";
    default: return "text-muted-foreground";
  }
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 80) return "text-chart-3";
  if (confidence >= 60) return "text-chart-2";
  return "text-chart-1";
};

export default function DiagnosticResults() {
  const [selectedDiagnosis, setSelectedDiagnosis] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Fetch diagnostic history
  const { data: historyData, isLoading, error, refetch } = useQuery({
    queryKey: ['diagnostic-history'],
    queryFn: fetchDiagnosticHistory,
    refetchInterval: 30000, // Refresh every 30 seconds
  });
  
  const savedAnalyses = historyData?.analyses || [];

  const handleReanalyze = () => {
    setIsAnalyzing(true);
    console.log("Starting diagnostic re-analysis...");
    // Simulate analysis time
    setTimeout(() => {
      setIsAnalyzing(false);
      console.log("Diagnostic analysis completed");
    }, 3000);
  };

  const toggleDetails = (index: number) => {
    setSelectedDiagnosis(selectedDiagnosis === index ? null : index);
    console.log(`Toggled details for diagnosis ${index}`);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              AI Diagnostic Analysis
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-muted-foreground">
                <Clock className="h-3 w-3 mr-1" />
                Analysis completed 2 min ago
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReanalyze}
                disabled={isAnalyzing}
                data-testid="button-reanalyze"
                className="hover-elevate"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-pulse-soft h-3 w-3 mr-2 bg-primary rounded-full" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <TrendingUp className="h-3 w-3 mr-2" />
                    Re-analyze
                  </>
                )}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              <div className="text-center">
                <Brain className="h-8 w-8 mx-auto mb-2 opacity-50 animate-pulse" />
                <p>Loading diagnostic history...</p>
              </div>
            </div>
          ) : savedAnalyses.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              <div className="text-center">
                <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No analyses available</p>
                <p className="text-sm mt-1">Upload a medical document to start analysis</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Recent Medical Analyses</h3>
                <Badge variant="outline">{savedAnalyses.length} total</Badge>
              </div>
              {savedAnalyses.map((analysis: any, index: number) => (
                <Card
                  key={analysis.id}
                  className={`transition-all hover-elevate cursor-pointer ${
                    selectedDiagnosis === index ? "ring-2 ring-primary" : ""
                  }`}
                  onClick={() => toggleDetails(index)}
                  data-testid={`analysis-card-${index}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <FileText className="h-5 w-5 text-primary" />
                        <div>
                          <h3 className="font-semibold text-lg">{analysis.fileName}</h3>
                          <p className="text-sm text-muted-foreground flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {new Date(analysis.createdAt).toLocaleDateString()} at {new Date(analysis.createdAt).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">
                          {analysis.processingTimeMs ? `${(analysis.processingTimeMs / 1000).toFixed(1)}s` : 'Processed'}
                        </Badge>
                        <Button
                          variant="ghost"
                          size="icon"
                          data-testid={`button-details-${index}`}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {selectedDiagnosis === index && (
                      <div className="animate-fade-in">
                        <div className="border-t pt-4 mt-3 space-y-4">
                          <div>
                            <h4 className="font-medium mb-2 text-primary">Intake Analysis</h4>
                            <p className="text-sm text-muted-foreground">{analysis.intake}</p>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2 text-primary">Medical Analysis</h4>
                            <p className="text-sm text-muted-foreground">{analysis.analysis}</p>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2 text-destructive">Triage Assessment</h4>
                            <p className="text-sm text-muted-foreground">{analysis.triage}</p>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2 text-chart-3">Explanation</h4>
                            <p className="text-sm text-muted-foreground">{analysis.explanation}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}