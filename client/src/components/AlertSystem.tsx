import { useState } from "react";
import { AlertTriangle, Bell, X, Clock, CheckCircle, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface MedicalAlert {
  id: string;
  type: "critical" | "warning" | "info";
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
  action?: string;
}

// Real alerts will be generated from AI analysis
const emptyAlerts: MedicalAlert[] = [];

const getAlertIcon = (type: string) => {
  switch (type) {
    case "critical": return <AlertTriangle className="h-4 w-4" />;
    case "warning": return <Bell className="h-4 w-4" />;
    case "info": return <CheckCircle className="h-4 w-4" />;
    default: return <Bell className="h-4 w-4" />;
  }
};

const getAlertColor = (type: string) => {
  switch (type) {
    case "critical": return "border-l-destructive bg-destructive/5 text-destructive";
    case "warning": return "border-l-chart-2 bg-orange-50 dark:bg-orange-950/20 text-chart-2";
    case "info": return "border-l-primary bg-primary/5 text-primary";
    default: return "border-l-muted";
  }
};

const getBadgeVariant = (type: string) => {
  switch (type) {
    case "critical": return "destructive";
    case "warning": return "secondary";
    case "info": return "outline";
    default: return "outline";
  }
};

export default function AlertSystem() {
  const [alerts, setAlerts] = useState(emptyAlerts);

  const handleAcknowledge = (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    );
    console.log(`Alert ${alertId} acknowledged`);
  };

  const handleDismiss = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    console.log(`Alert ${alertId} dismissed`);
  };

  const handleAction = (alertId: string, action: string) => {
    console.log(`Executing action for alert ${alertId}: ${action}`);
    handleAcknowledge(alertId);
  };

  const criticalAlerts = alerts.filter(alert => alert.type === "critical" && !alert.acknowledged);
  const activeAlerts = alerts.filter(alert => !alert.acknowledged);

  return (
    <div className="space-y-6">
      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <Alert className="border-destructive bg-destructive/10 animate-pulse-soft">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          <AlertDescription className="text-destructive font-medium">
            {criticalAlerts.length} critical alert{criticalAlerts.length > 1 ? 's' : ''} requiring immediate attention
          </AlertDescription>
        </Alert>
      )}

      {/* Alert Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Alert Center
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" data-testid="active-alerts-count">
                {activeAlerts.length} Active
              </Badge>
              <Badge variant="secondary" data-testid="total-alerts-count">
                {alerts.length} Total
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {alerts.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No active alerts</p>
                <p className="text-sm mt-2">Alerts will appear here from AI analysis</p>
              </div>
            ) : (
              alerts.map((alert) => (
                <Card
                  key={alert.id}
                  className={`border-l-4 transition-all ${getAlertColor(alert.type)} ${
                    alert.acknowledged ? "opacity-60" : "animate-fade-in"
                  }`}
                  data-testid={`alert-${alert.id}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-3 flex-1">
                        <div className="mt-0.5">
                          {getAlertIcon(alert.type)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold">{alert.title}</h3>
                            <Badge 
                              variant={getBadgeVariant(alert.type)}
                              data-testid={`badge-type-${alert.id}`}
                            >
                              {alert.type.toUpperCase()}
                            </Badge>
                            {alert.acknowledged && (
                              <Badge variant="outline" className="text-chart-3">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                Acknowledged
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm mb-2">{alert.description}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            {alert.timestamp}
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col gap-2">
                        {!alert.acknowledged && (
                          <>
                            {alert.action && (
                              <Button
                                size="sm"
                                variant="default"
                                onClick={() => handleAction(alert.id, alert.action!)}
                                data-testid={`button-action-${alert.id}`}
                                className="text-xs hover-elevate"
                              >
                                <Zap className="h-3 w-3 mr-1" />
                                {alert.action}
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleAcknowledge(alert.id)}
                              data-testid={`button-acknowledge-${alert.id}`}
                              className="text-xs hover-elevate"
                            >
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Acknowledge
                            </Button>
                          </>
                        )}
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDismiss(alert.id)}
                          data-testid={`button-dismiss-${alert.id}`}
                          className="text-xs hover-elevate"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}