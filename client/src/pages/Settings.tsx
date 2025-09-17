import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings as SettingsIcon, User, Shield, Bell, Database, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";

export default function Settings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const clearDataMutation = useMutation({
    mutationFn: () => apiRequest('DELETE', '/api/clear-all-data'),
    onSuccess: () => {
      toast({
        title: "Data cleared successfully",
        description: "All medical analyses and user data have been removed.",
      });
      // Invalidate all queries to refresh the UI
      queryClient.invalidateQueries();
    },
    onError: () => {
      toast({
        title: "Error clearing data",
        description: "Failed to clear data. Please try again.",
        variant: "destructive",
      });
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure system preferences and user settings
        </p>
      </div>
      
      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Profile Settings
            </CardTitle>
            <CardDescription>
              Manage your personal profile and preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="email-notifications">Email notifications</Label>
              <Switch id="email-notifications" />
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="auto-save">Auto-save forms</Label>
              <Switch id="auto-save" defaultChecked />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Security Settings
            </CardTitle>
            <CardDescription>
              Configure security and access controls
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="two-factor">Two-factor authentication</Label>
              <Switch id="two-factor" />
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="session-timeout">Auto session timeout</Label>
              <Switch id="session-timeout" defaultChecked />
            </div>
            <Button variant="outline">Change Password</Button>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Alert Preferences
            </CardTitle>
            <CardDescription>
              Configure alert thresholds and notifications
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="high-risk-alerts">High-risk patient alerts</Label>
              <Switch id="high-risk-alerts" defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="system-alerts">System maintenance alerts</Label>
              <Switch id="system-alerts" defaultChecked />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Management
            </CardTitle>
            <CardDescription>
              Manage data retention and export options
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button variant="outline" data-testid="button-export-data">Export Patient Data</Button>
            <Button variant="outline" data-testid="button-backup-data">Backup System Data</Button>
            <Button variant="destructive" data-testid="button-clear-cache">Clear Cache</Button>
            
            <div className="pt-4 border-t">
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button 
                    variant="destructive" 
                    className="w-full"
                    data-testid="button-clear-all-data"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear All Data
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. This will permanently delete all medical analyses, 
                      patient data, and user accounts from the database. This is irreversible.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel data-testid="button-cancel-clear">Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={() => clearDataMutation.mutate()}
                      disabled={clearDataMutation.isPending}
                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                      data-testid="button-confirm-clear"
                    >
                      {clearDataMutation.isPending ? "Clearing..." : "Yes, clear all data"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}