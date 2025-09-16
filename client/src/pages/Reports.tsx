import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Download, Calendar, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Reports() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Reports</h1>
        <p className="text-muted-foreground">
          Generate and view medical reports and analytics
        </p>
      </div>
      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Patient Reports
            </CardTitle>
            <CardDescription>
              Individual patient diagnostic reports
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button className="w-full">
              <Download className="h-4 w-4 mr-2" />
              Generate Report
            </Button>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Monthly Summary
            </CardTitle>
            <CardDescription>
              Monthly diagnostic statistics and trends
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button className="w-full" variant="outline">
              <Download className="h-4 w-4 mr-2" />
              View Summary
            </Button>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Custom Reports
            </CardTitle>
            <CardDescription>
              Create custom filtered reports
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button className="w-full" variant="outline">
              Create Custom
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}