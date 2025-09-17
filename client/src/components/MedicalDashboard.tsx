import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Activity, Users, Clock, TrendingUp, Brain, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Real-time data - will be populated from actual patient analyses
const emptyStats = {
  totalPatients: 0,
  activeAlerts: 0,
  avgProcessingTime: "--",
  accuracyRate: "--"
};

const emptyDiagnosticTrends: any[] = [];
const emptyConditionDistribution: any[] = [];
const emptyRecentActivity: any[] = [];

const getRiskColor = (risk: string) => {
  switch (risk) {
    case "high": return "text-destructive";
    case "medium": return "text-chart-2";
    case "low": return "text-chart-3";
    default: return "text-muted-foreground";
  }
};

const getRiskBadge = (risk: string) => {
  switch (risk) {
    case "high": return "destructive";
    case "medium": return "secondary";
    case "low": return "outline";
    default: return "outline";
  }
};

export default function MedicalDashboard() {
  const [timeRange, setTimeRange] = useState("6m");

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="hover-elevate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="stat-total-patients">
              {emptyStats.totalPatients.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              No analysis performed yet
            </p>
          </CardContent>
        </Card>

        <Card className="hover-elevate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-destructive" data-testid="stat-active-alerts">
              {emptyStats.activeAlerts}
            </div>
            <p className="text-xs text-muted-foreground">
              No active alerts
            </p>
          </CardContent>
        </Card>

        <Card className="hover-elevate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="stat-processing-time">
              {emptyStats.avgProcessingTime}
            </div>
            <p className="text-xs text-muted-foreground">
              Ready for analysis
            </p>
          </CardContent>
        </Card>

        <Card className="hover-elevate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AI Accuracy Rate</CardTitle>
            <Brain className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-chart-3" data-testid="stat-accuracy-rate">
              {emptyStats.accuracyRate}
            </div>
            <p className="text-xs text-muted-foreground">
              Awaiting patient data
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Diagnostic Trends */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Diagnostic Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-[300px] text-muted-foreground">
              <div className="text-center">
                <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No diagnostic data available yet</p>
                <p className="text-sm mt-2">Start analyzing patient documents to see trends</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Condition Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Condition Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-[300px] text-muted-foreground">
              <div className="text-center">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No condition data available yet</p>
                <p className="text-sm mt-2">Upload medical documents to see condition distribution</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-primary" />
            Recent Diagnostic Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            <div className="text-center">
              <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No recent activity</p>
              <p className="text-sm mt-1">Patient analyses will appear here</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}