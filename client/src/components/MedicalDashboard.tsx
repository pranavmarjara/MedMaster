import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Activity, Users, Clock, TrendingUp, Brain, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

//TODO: remove mock functionality - replace with real API data
const mockStats = {
  totalPatients: 847,
  activeAlerts: 3,
  avgProcessingTime: "2.4 min",
  accuracyRate: "94.2%"
};

const mockDiagnosticTrends = [
  { month: "Jan", diagnoses: 156, accuracy: 92 },
  { month: "Feb", diagnoses: 178, accuracy: 94 },
  { month: "Mar", diagnoses: 195, accuracy: 96 },
  { month: "Apr", diagnoses: 203, accuracy: 95 },
  { month: "May", diagnoses: 187, accuracy: 94 },
  { month: "Jun", diagnoses: 221, accuracy: 97 },
];

const mockConditionDistribution = [
  { name: "Respiratory Infections", value: 35, color: "hsl(var(--chart-1))" },
  { name: "Cardiovascular", value: 20, color: "hsl(var(--chart-2))" },
  { name: "Musculoskeletal", value: 18, color: "hsl(var(--chart-3))" },
  { name: "Gastrointestinal", value: 15, color: "hsl(var(--chart-4))" },
  { name: "Other", value: 12, color: "hsl(var(--chart-5))" },
];

const mockRecentActivity = [
  { id: 1, patient: "Patient #4851", condition: "Acute Bronchitis", confidence: 89, timestamp: "2 min ago", risk: "low" },
  { id: 2, patient: "Patient #4852", condition: "Hypertensive Crisis", confidence: 94, timestamp: "5 min ago", risk: "high" },
  { id: 3, patient: "Patient #4853", condition: "Migraine", confidence: 78, timestamp: "8 min ago", risk: "medium" },
  { id: 4, patient: "Patient #4854", condition: "Gastroenteritis", confidence: 85, timestamp: "12 min ago", risk: "low" },
];

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
              {mockStats.totalPatients.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              +12% from last month
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
              {mockStats.activeAlerts}
            </div>
            <p className="text-xs text-muted-foreground">
              Requiring immediate attention
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
              {mockStats.avgProcessingTime}
            </div>
            <p className="text-xs text-muted-foreground">
              -8% improvement this week
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
              {mockStats.accuracyRate}
            </div>
            <p className="text-xs text-muted-foreground">
              Based on clinician feedback
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
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={mockDiagnosticTrends}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="month" />
                <YAxis />
                <Bar dataKey="diagnoses" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
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
            <div className="flex items-center justify-center">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={mockConditionDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {mockConditionDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-1 gap-2 mt-4">
              {mockConditionDistribution.map((item, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span>{item.name}</span>
                  <span className="text-muted-foreground ml-auto">{item.value}%</span>
                </div>
              ))}
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
          <div className="space-y-3">
            {mockRecentActivity.map((activity) => (
              <div
                key={activity.id}
                className="flex items-center justify-between p-3 rounded-md border hover-elevate transition-colors"
                data-testid={`activity-${activity.id}`}
              >
                <div className="flex items-center gap-3">
                  <div>
                    <div className="font-medium text-sm">{activity.patient}</div>
                    <div className="text-sm text-muted-foreground">{activity.condition}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-sm font-medium" data-testid={`confidence-${activity.id}`}>
                      {activity.confidence}%
                    </div>
                    <div className="text-xs text-muted-foreground">{activity.timestamp}</div>
                  </div>
                  <Badge 
                    variant={getRiskBadge(activity.risk)}
                    className={getRiskColor(activity.risk)}
                    data-testid={`risk-${activity.id}`}
                  >
                    {activity.risk.toUpperCase()}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}