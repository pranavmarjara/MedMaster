import { AlertTriangle, Shield, TrendingUp, Heart, Thermometer, Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface RiskFactor {
  name: string;
  value: string;
  severity: "low" | "medium" | "high";
  impact: number;
}

//TODO: remove mock functionality - replace with real API data
const mockRiskFactors: RiskFactor[] = [
  { name: "Cardiovascular Risk", value: "Moderate", severity: "medium", impact: 65 },
  { name: "Infection Severity", value: "Mild-Moderate", severity: "medium", impact: 58 },
  { name: "Respiratory Compromise", value: "Low", severity: "low", impact: 25 },
  { name: "Dehydration Risk", value: "Low", severity: "low", impact: 20 },
];

const overallRiskScore = 42; // TODO: calculate from actual data
const riskLevel = overallRiskScore >= 70 ? "high" : overallRiskScore >= 40 ? "medium" : "low";

const getRiskIcon = (level: string) => {
  switch (level) {
    case "high": return <AlertTriangle className="h-5 w-5 text-destructive" />;
    case "medium": return <TrendingUp className="h-5 w-5 text-chart-2" />;
    case "low": return <Shield className="h-5 w-5 text-chart-3" />;
    default: return <Shield className="h-5 w-5" />;
  }
};

const getRiskColor = (level: string) => {
  switch (level) {
    case "high": return "text-destructive border-destructive bg-destructive/5";
    case "medium": return "text-chart-2 border-chart-2 bg-orange-50 dark:bg-orange-950/20";
    case "low": return "text-chart-3 border-chart-3 bg-green-50 dark:bg-green-950/20";
    default: return "text-muted-foreground";
  }
};

const getFactorIcon = (name: string) => {
  if (name.includes("Cardiovascular")) return <Heart className="h-4 w-4" />;
  if (name.includes("Respiratory")) return <Activity className="h-4 w-4" />;
  if (name.includes("Infection")) return <Thermometer className="h-4 w-4" />;
  return <TrendingUp className="h-4 w-4" />;
};

export default function RiskAssessment() {
  return (
    <div className="space-y-6">
      {/* Overall Risk Score */}
      <Card className={`border-2 ${getRiskColor(riskLevel)}`}>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {getRiskIcon(riskLevel)}
              Overall Risk Assessment
            </div>
            <Badge 
              variant="outline" 
              className={getRiskColor(riskLevel)}
              data-testid="badge-overall-risk"
            >
              {riskLevel.toUpperCase()} RISK
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2" data-testid="risk-score">
                {overallRiskScore}
              </div>
              <div className="text-sm text-muted-foreground">Risk Score (0-100)</div>
            </div>
            <Progress 
              value={overallRiskScore} 
              className="h-3"
              data-testid="progress-overall-risk"
            />
            <div className="text-sm text-muted-foreground text-center">
              {riskLevel === "high" && "Immediate medical attention recommended"}
              {riskLevel === "medium" && "Close monitoring and follow-up care advised"}
              {riskLevel === "low" && "Standard care protocols applicable"}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Factors Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Risk Factors Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mockRiskFactors.map((factor, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-md border hover-elevate"
                data-testid={`risk-factor-${index}`}
              >
                <div className="flex items-center gap-3">
                  <div className={getRiskColor(factor.severity).split(' ')[0]}>
                    {getFactorIcon(factor.name)}
                  </div>
                  <div>
                    <h4 className="font-medium">{factor.name}</h4>
                    <p className="text-sm text-muted-foreground">{factor.value}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-sm font-medium" data-testid={`impact-score-${index}`}>
                      {factor.impact}%
                    </div>
                    <div className="text-xs text-muted-foreground">Impact</div>
                  </div>
                  <Badge 
                    variant="outline" 
                    className={getRiskColor(factor.severity)}
                    data-testid={`severity-badge-${index}`}
                  >
                    {factor.severity}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            Clinical Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 rounded-md bg-muted/50">
              <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Monitor vital signs every 4 hours</p>
                <p className="text-xs text-muted-foreground">Due to moderate cardiovascular risk</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 rounded-md bg-muted/50">
              <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Administer antibiotics if bacterial infection suspected</p>
                <p className="text-xs text-muted-foreground">Based on infection severity assessment</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 rounded-md bg-muted/50">
              <div className="h-2 w-2 bg-primary rounded-full mt-2 flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Schedule follow-up in 24-48 hours</p>
                <p className="text-xs text-muted-foreground">To reassess condition progression</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}