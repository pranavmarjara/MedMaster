import { useState } from "react";
import { Brain, TrendingUp, Eye, Clock, CheckCircle } from "lucide-react";
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
          <div className="space-y-4">
            {emptyDiagnoses.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                <div className="text-center">
                  <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No diagnosis available</p>
                  <p className="text-sm mt-1">Upload a medical document to start analysis</p>
                </div>
              </div>
            ) : (
              emptyDiagnoses.map((diagnosis, index) => (
              <Card
                key={index}
                className={`transition-all hover-elevate cursor-pointer ${
                  selectedDiagnosis === index ? "ring-2 ring-primary" : ""
                }`}
                onClick={() => toggleDetails(index)}
                data-testid={`diagnosis-card-${index}`}
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-lg">{diagnosis.condition}</h3>
                      <Badge 
                        variant="outline" 
                        className={getSeverityColor(diagnosis.severity)}
                        data-testid={`badge-severity-${index}`}
                      >
                        {diagnosis.severity.toUpperCase()}
                      </Badge>
                      <Badge variant="secondary" data-testid={`badge-icd10-${index}`}>
                        {diagnosis.icd10}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span 
                        className={`text-sm font-medium ${getConfidenceColor(diagnosis.confidence)}`}
                        data-testid={`confidence-${index}`}
                      >
                        {diagnosis.confidence}%
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        data-testid={`button-details-${index}`}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-muted-foreground mb-1">
                      <span>Confidence Level</span>
                      <span>{diagnosis.confidence}%</span>
                    </div>
                    <Progress 
                      value={diagnosis.confidence} 
                      className="h-2"
                      data-testid={`progress-confidence-${index}`}
                    />
                  </div>

                  {selectedDiagnosis === index && (
                    <div className="animate-fade-in">
                      <div className="border-t pt-3 mt-3">
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-chart-3" />
                          Supporting Evidence
                        </h4>
                        <ul className="space-y-1">
                          {diagnosis.evidence.map((evidence, evidenceIndex) => (
                            <li 
                              key={evidenceIndex}
                              className="text-sm text-muted-foreground flex items-center gap-2"
                              data-testid={`evidence-${index}-${evidenceIndex}`}
                            >
                              <div className="h-1.5 w-1.5 bg-primary rounded-full flex-shrink-0" />
                              {evidence}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}