import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Activity, Users, Clock, TrendingUp, Brain, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Fetch real dashboard data from API
const fetchDashboardStats = async () => {
  const response = await fetch('/api/dashboard-stats');
  if (!response.ok) {
    throw new Error('Failed to fetch dashboard stats');
  }
  return response.json();
};

// Fallback data structure
const emptyStats = {
  totalPatients: 0,
  activeAlerts: 0,
  avgProcessingTime: "--",
  accuracyRate: "--"
};

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
  
  // Fetch dashboard statistics
  const { data: dashboardData, isLoading, error } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: fetchDashboardStats,
    refetchInterval: 30000, // Refresh every 30 seconds
  });
  
  const stats = dashboardData?.stats || emptyStats;
  const recentActivity = dashboardData?.recentActivity || [];

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
              {stats.totalPatients.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.totalPatients === 0 ? "No analysis performed yet" : "Total medical analyses"}
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
              {stats.activeAlerts}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.activeAlerts === 0 ? "No active alerts" : "High-priority cases"}
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
              {stats.avgProcessingTime}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.avgProcessingTime === "--" ? "Ready for analysis" : "Average analysis time"}
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
              {stats.accuracyRate}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.accuracyRate === "--" ? "Awaiting patient data" : "AI diagnostic accuracy"}
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
            {stats.totalPatients === 0 ? (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No diagnostic data available yet</p>
                  <p className="text-sm mt-2">Start analyzing patient documents to see trends</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Diagnostic trends visualization</p>
                  <p className="text-sm mt-2">Based on {stats.totalPatients} analyses</p>
                  <p className="text-xs mt-2 text-chart-3">Chart visualization coming soon</p>
                </div>
              </div>
            )}
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
            {stats.totalPatients === 0 ? (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No condition data available yet</p>
                  <p className="text-sm mt-2">Upload medical documents to see condition distribution</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Condition distribution analysis</p>
                  <p className="text-sm mt-2">Based on {stats.totalPatients} medical analyses</p>
                  <p className="text-xs mt-2 text-chart-3">Detailed distribution charts coming soon</p>
                </div>
              </div>
            )}
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
          {recentActivity.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              <div className="text-center">
                <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No recent activity</p>
                <p className="text-sm mt-1">Patient analyses will appear here</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {recentActivity.slice(0, 5).map((analysis: any, index: number) => (
                <div key={analysis.id} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 bg-primary rounded-full" />
                    <div>
                      <p className="font-medium text-sm">{analysis.fileName}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(analysis.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {analysis.processingTimeMs ? `${(analysis.processingTimeMs / 1000).toFixed(1)}s` : 'Processed'}
                  </Badge>
                </div>
              ))}
              {recentActivity.length > 5 && (
                <p className="text-xs text-muted-foreground text-center pt-2">
                  And {recentActivity.length - 5} more analyses...
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}