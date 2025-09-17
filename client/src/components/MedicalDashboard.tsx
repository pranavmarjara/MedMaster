import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Activity, Users, Clock, TrendingUp, Brain, AlertTriangle, FileText, Eye } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";

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
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Recent Diagnoses</h3>
                <Link href="/diagnostics">
                  <Button variant="outline" size="sm" className="hover-elevate">
                    <Eye className="h-4 w-4 mr-2" />
                    View All Results
                  </Button>
                </Link>
              </div>
              {recentActivity.slice(0, 3).map((analysis: any, index: number) => (
                <Card key={analysis.id} className="hover-elevate transition-all">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <FileText className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-semibold text-sm">{analysis.fileName}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(analysis.createdAt).toLocaleDateString()} at {new Date(analysis.createdAt).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant="secondary" className="text-xs">
                        {analysis.processingTimeMs ? `${(analysis.processingTimeMs / 1000).toFixed(1)}s` : 'Processed'}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      {analysis.triage && (
                        <div>
                          <span className="font-medium text-red-600">Triage: </span>
                          <span className="text-muted-foreground">
                            {analysis.triage.length > 100 ? `${analysis.triage.substring(0, 100)}...` : analysis.triage}
                          </span>
                        </div>
                      )}
                      {analysis.analysis && (
                        <div>
                          <span className="font-medium text-purple-600">Analysis: </span>
                          <span className="text-muted-foreground">
                            {analysis.analysis.length > 100 ? `${analysis.analysis.substring(0, 100)}...` : analysis.analysis}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    <Link href="/diagnostics">
                      <Button variant="ghost" size="sm" className="mt-3 text-xs">
                        <Eye className="h-3 w-3 mr-1" />
                        View Full Results
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              ))}
              {recentActivity.length > 3 && (
                <div className="text-center pt-2">
                  <Link href="/diagnostics">
                    <Button variant="outline" size="sm">
                      View {recentActivity.length - 3} More Diagnoses
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}