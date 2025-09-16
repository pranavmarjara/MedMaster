import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, Plus, Search, UserCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function Patients() {
  const recentPatients = [
    { id: 1, name: "John Smith", lastVisit: "2024-01-15", status: "Active" },
    { id: 2, name: "Sarah Johnson", lastVisit: "2024-01-14", status: "Follow-up" },
    { id: 3, name: "Michael Brown", lastVisit: "2024-01-12", status: "Completed" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patients</h1>
          <p className="text-muted-foreground">
            Manage patient records and information
          </p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Add Patient
        </Button>
      </div>
      
      <div className="flex items-center space-x-2">
        <Search className="h-4 w-4" />
        <Input
          placeholder="Search patients..."
          className="max-w-sm"
        />
      </div>
      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Total Patients
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">247</div>
            <p className="text-xs text-muted-foreground">
              +12 from last month
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserCheck className="h-5 w-5" />
              Active Cases
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">89</div>
            <p className="text-xs text-muted-foreground">
              Requiring follow-up
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Recent Patients</CardTitle>
          <CardDescription>
            Recently registered or updated patient records
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentPatients.map((patient) => (
              <div key={patient.id} className="flex items-center justify-between border-b pb-2">
                <div>
                  <p className="font-medium">{patient.name}</p>
                  <p className="text-sm text-muted-foreground">Last visit: {patient.lastVisit}</p>
                </div>
                <div className="text-sm">
                  <span className="px-2 py-1 rounded-full bg-primary/10 text-primary">
                    {patient.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}