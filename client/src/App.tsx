import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import MedicalSidebar from "@/components/MedicalSidebar";
import ThemeToggle from "@/components/ThemeToggle";
import Dashboard from "@/pages/Dashboard";
import Intake from "@/pages/Intake";
import Diagnostics from "@/pages/Diagnostics";
import Risk from "@/pages/Risk";
import Alerts from "@/pages/Alerts";
import Reports from "@/pages/Reports";
import Patients from "@/pages/Patients";
import Settings from "@/pages/Settings";
import Explainability from "@/pages/Explainability";
import NotFound from "@/pages/not-found";
import { Stethoscope } from "lucide-react";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/intake" component={Intake} />
      <Route path="/diagnostics" component={Diagnostics} />
      <Route path="/risk" component={Risk} />
      <Route path="/alerts" component={Alerts} />
      <Route path="/reports" component={Reports} />
      <Route path="/patients" component={Patients} />
      <Route path="/settings" component={Settings} />
      <Route path="/explainability" component={Explainability} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const style = {
    "--sidebar-width": "20rem",
    "--sidebar-width-icon": "4rem",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SidebarProvider style={style as React.CSSProperties}>
          <div className="flex h-screen w-full">
            <MedicalSidebar />
            <div className="flex flex-col flex-1">
              <header className="flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <div className="flex items-center gap-3">
                  <SidebarTrigger data-testid="button-sidebar-toggle" />
                  <div className="flex items-center gap-2">
                    <Stethoscope className="h-6 w-6 text-primary" />
                    <h1 className="text-xl font-semibold text-primary">MedAI Triage</h1>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="text-sm text-muted-foreground hidden sm:block">
                    AI-Powered Diagnostic Assistant
                  </div>
                  <ThemeToggle />
                </div>
              </header>
              <main className="flex-1 overflow-auto p-6 bg-background">
                <div className="animate-fade-in">
                  <Router />
                </div>
              </main>
            </div>
          </div>
        </SidebarProvider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
