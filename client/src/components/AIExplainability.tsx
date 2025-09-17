import { useState } from "react";
import { Brain, Eye, FileText, Lightbulb, ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface ReasoningStep {
  step: number;
  description: string;
  evidence: string[];
  confidence: number;
  weight: number;
}

interface DataPoint {
  category: string;
  value: string;
  relevance: "high" | "medium" | "low";
  impact: number;
}

// Real reasoning data will be populated from AI analysis
const emptyReasoningSteps: ReasoningStep[] = [];
const emptyDataPoints: DataPoint[] = [];

const getRelevanceColor = (relevance: string) => {
  switch (relevance) {
    case "high": return "text-chart-1 bg-red-50 dark:bg-red-950/20";
    case "medium": return "text-chart-2 bg-orange-50 dark:bg-orange-950/20";
    case "low": return "text-chart-3 bg-green-50 dark:bg-green-950/20";
    default: return "text-muted-foreground";
  }
};

export default function AIExplainability() {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([1]));
  const [showDataMapping, setShowDataMapping] = useState(false);

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps(prev => {
      const newSet = new Set(prev);
      if (newSet.has(stepNumber)) {
        newSet.delete(stepNumber);
      } else {
        newSet.add(stepNumber);
      }
      return newSet;
    });
    console.log(`Toggled reasoning step ${stepNumber}`);
  };

  const toggleDataMapping = () => {
    setShowDataMapping(!showDataMapping);
    console.log(`Data mapping ${showDataMapping ? 'hidden' : 'shown'}`);
  };

  return (
    <div className="space-y-6">
      {/* AI Reasoning Process */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Reasoning Process
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Transparent breakdown of how the AI reached its diagnostic conclusions
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {emptyReasoningSteps.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                <div className="text-center">
                  <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No reasoning analysis available</p>
                  <p className="text-sm mt-1">AI reasoning steps will appear here after analysis</p>
                </div>
              </div>
            ) : (
              emptyReasoningSteps.map((step) => (
              <Card key={step.step} className="hover-elevate">
                <Collapsible
                  open={expandedSteps.has(step.step)}
                  onOpenChange={() => toggleStep(step.step)}
                >
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground text-sm font-medium">
                            {step.step}
                          </div>
                          <div>
                            <h3 className="font-medium text-left">{step.description}</h3>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" data-testid={`confidence-${step.step}`}>
                                {step.confidence}% Confidence
                              </Badge>
                              <Badge variant="secondary" data-testid={`weight-${step.step}`}>
                                {step.weight}% Weight
                              </Badge>
                            </div>
                          </div>
                        </div>
                        <Button variant="ghost" size="icon" data-testid={`toggle-step-${step.step}`}>
                          {expandedSteps.has(step.step) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="pt-0 animate-fade-in">
                      <div className="ml-11">
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          <Eye className="h-4 w-4 text-primary" />
                          Evidence Considered
                        </h4>
                        <ul className="space-y-1">
                          {step.evidence.map((evidence, index) => (
                            <li 
                              key={index}
                              className="text-sm text-muted-foreground flex items-center gap-2"
                              data-testid={`evidence-${step.step}-${index}`}
                            >
                              <div className="h-1.5 w-1.5 bg-primary rounded-full flex-shrink-0" />
                              {evidence}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Data Point Mapping */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Data Point Impact Analysis
            </div>
            <Button
              variant="outline"
              onClick={toggleDataMapping}
              data-testid="button-toggle-data-mapping"
              className="hover-elevate"
            >
              {showDataMapping ? "Hide Details" : "Show Details"}
            </Button>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            How each data point influenced the diagnostic reasoning
          </p>
        </CardHeader>
        {showDataMapping && (
          <CardContent className="animate-fade-in">
            <div className="grid gap-3">
              {emptyDataPoints.length === 0 ? (
                <div className="flex items-center justify-center h-32 text-muted-foreground">
                  <div className="text-center">
                    <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No data point analysis available</p>
                    <p className="text-sm mt-1">Data impact analysis will appear here</p>
                  </div>
                </div>
              ) : (
                emptyDataPoints
                  .sort((a, b) => b.impact - a.impact)
                  .map((dataPoint, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-md border hover-elevate"
                  data-testid={`data-point-${index}`}
                >
                  <div className="flex items-center gap-3">
                    <div className="text-sm font-medium">{dataPoint.category}</div>
                    <div className="text-sm text-muted-foreground">{dataPoint.value}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-sm font-medium" data-testid={`impact-${index}`}>
                        {dataPoint.impact}%
                      </div>
                      <div className="text-xs text-muted-foreground">Impact</div>
                    </div>
                    <Badge 
                      variant="outline" 
                      className={getRelevanceColor(dataPoint.relevance)}
                      data-testid={`relevance-${index}`}
                    >
                      {dataPoint.relevance}
                    </Badge>
                  </div>
                </div>
                ))
              )}
            </div>
          </CardContent>
        )}
      </Card>

      {/* Trust & Validation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-primary" />
            Trust & Validation Notes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="p-3 rounded-md bg-muted/50">
              <p className="text-sm font-medium">Model Validation</p>
              <p className="text-xs text-muted-foreground mt-1">
                This AI model has been trained on validated medical datasets and cross-referenced with clinical guidelines.
              </p>
            </div>
            <div className="p-3 rounded-md bg-muted/50">
              <p className="text-sm font-medium">Human Oversight Required</p>
              <p className="text-xs text-muted-foreground mt-1">
                All AI recommendations must be reviewed and validated by qualified medical professionals before clinical implementation.
              </p>
            </div>
            <div className="p-3 rounded-md bg-muted/50">
              <p className="text-sm font-medium">Uncertainty Acknowledgment</p>
              <p className="text-xs text-muted-foreground mt-1">
                Areas of diagnostic uncertainty are clearly flagged and require additional clinical investigation.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}