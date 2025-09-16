import { useState } from "react";
import { ChevronRight, User, FileText, Activity, Clipboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";

const steps = [
  { id: 1, title: "Patient Info", icon: User },
  { id: 2, title: "Chief Complaint", icon: FileText },
  { id: 3, title: "Vitals", icon: Activity },
  { id: 4, title: "Review", icon: Clipboard },
];

interface PatientData {
  // Basic info
  firstName: string;
  lastName: string;
  age: string;
  gender: string;
  // Chief complaint
  chiefComplaint: string;
  symptomDuration: string;
  painLevel: string;
  // Vitals
  bloodPressure: string;
  heartRate: string;
  temperature: string;
  oxygenSaturation: string;
}

export default function PatientIntakeForm() {
  const [currentStep, setCurrentStep] = useState(1);
  const [patientData, setPatientData] = useState<PatientData>({
    firstName: "",
    lastName: "",
    age: "",
    gender: "",
    chiefComplaint: "",
    symptomDuration: "",
    painLevel: "",
    bloodPressure: "",
    heartRate: "",
    temperature: "",
    oxygenSaturation: "",
  });

  const progress = (currentStep / steps.length) * 100;

  const handleNext = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
      console.log(`Advanced to step ${currentStep + 1}`);
    } else {
      console.log("Form submitted", patientData);
      // TODO: Submit to backend
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      console.log(`Returned to step ${currentStep - 1}`);
    }
  };

  const updatePatientData = (field: keyof PatientData, value: string) => {
    setPatientData(prev => ({ ...prev, [field]: value }));
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="firstName">First Name *</Label>
                <Input
                  id="firstName"
                  value={patientData.firstName}
                  onChange={(e) => updatePatientData("firstName", e.target.value)}
                  data-testid="input-first-name"
                />
              </div>
              <div>
                <Label htmlFor="lastName">Last Name *</Label>
                <Input
                  id="lastName"
                  value={patientData.lastName}
                  onChange={(e) => updatePatientData("lastName", e.target.value)}
                  data-testid="input-last-name"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="age">Age *</Label>
                <Input
                  id="age"
                  type="number"
                  value={patientData.age}
                  onChange={(e) => updatePatientData("age", e.target.value)}
                  data-testid="input-age"
                />
              </div>
              <div>
                <Label htmlFor="gender">Gender *</Label>
                <Select value={patientData.gender} onValueChange={(value) => updatePatientData("gender", value)}>
                  <SelectTrigger data-testid="select-gender">
                    <SelectValue placeholder="Select gender" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                    <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        );
      case 2:
        return (
          <div className="space-y-4 animate-fade-in">
            <div>
              <Label htmlFor="chiefComplaint">Chief Complaint *</Label>
              <Textarea
                id="chiefComplaint"
                placeholder="Describe the main reason for this visit..."
                value={patientData.chiefComplaint}
                onChange={(e) => updatePatientData("chiefComplaint", e.target.value)}
                data-testid="textarea-chief-complaint"
                className="min-h-24"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="symptomDuration">Symptom Duration</Label>
                <Select value={patientData.symptomDuration} onValueChange={(value) => updatePatientData("symptomDuration", value)}>
                  <SelectTrigger data-testid="select-duration">
                    <SelectValue placeholder="Select duration" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="acute">Less than 24 hours</SelectItem>
                    <SelectItem value="subacute">1-7 days</SelectItem>
                    <SelectItem value="chronic">More than 7 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="painLevel">Pain Level (0-10)</Label>
                <Select value={patientData.painLevel} onValueChange={(value) => updatePatientData("painLevel", value)}>
                  <SelectTrigger data-testid="select-pain-level">
                    <SelectValue placeholder="Select pain level" />
                  </SelectTrigger>
                  <SelectContent>
                    {[...Array(11)].map((_, i) => (
                      <SelectItem key={i} value={i.toString()}>{i} - {i === 0 ? "No pain" : i <= 3 ? "Mild" : i <= 6 ? "Moderate" : "Severe"}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        );
      case 3:
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="bloodPressure">Blood Pressure (mmHg)</Label>
                <Input
                  id="bloodPressure"
                  placeholder="120/80"
                  value={patientData.bloodPressure}
                  onChange={(e) => updatePatientData("bloodPressure", e.target.value)}
                  data-testid="input-blood-pressure"
                />
              </div>
              <div>
                <Label htmlFor="heartRate">Heart Rate (bpm)</Label>
                <Input
                  id="heartRate"
                  type="number"
                  placeholder="75"
                  value={patientData.heartRate}
                  onChange={(e) => updatePatientData("heartRate", e.target.value)}
                  data-testid="input-heart-rate"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="temperature">Temperature (°F)</Label>
                <Input
                  id="temperature"
                  type="number"
                  step="0.1"
                  placeholder="98.6"
                  value={patientData.temperature}
                  onChange={(e) => updatePatientData("temperature", e.target.value)}
                  data-testid="input-temperature"
                />
              </div>
              <div>
                <Label htmlFor="oxygenSaturation">Oxygen Saturation (%)</Label>
                <Input
                  id="oxygenSaturation"
                  type="number"
                  placeholder="98"
                  value={patientData.oxygenSaturation}
                  onChange={(e) => updatePatientData("oxygenSaturation", e.target.value)}
                  data-testid="input-oxygen-saturation"
                />
              </div>
            </div>
          </div>
        );
      case 4:
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="text-sm text-muted-foreground mb-4">
              Please review the entered information before proceeding with analysis.
            </div>
            <div className="space-y-3">
              <div>
                <strong>Patient:</strong> {patientData.firstName} {patientData.lastName}, {patientData.age} years old, {patientData.gender}
              </div>
              <div>
                <strong>Chief Complaint:</strong> {patientData.chiefComplaint}
              </div>
              <div>
                <strong>Duration:</strong> {patientData.symptomDuration} | <strong>Pain Level:</strong> {patientData.painLevel}/10
              </div>
              <div>
                <strong>Vitals:</strong> BP {patientData.bloodPressure}, HR {patientData.heartRate}, Temp {patientData.temperature}°F, O2 {patientData.oxygenSaturation}%
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clipboard className="h-5 w-5 text-primary" />
          Patient Intake Assessment
        </CardTitle>
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Step {currentStep} of {steps.length}</span>
            <span>{Math.round(progress)}% Complete</span>
          </div>
          <Progress value={progress} className="h-2" data-testid="progress-intake" />
        </div>
        <div className="flex gap-1 mt-4">
          {steps.map((step) => (
            <div
              key={step.id}
              className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors ${
                currentStep === step.id
                  ? "bg-primary text-primary-foreground"
                  : currentStep > step.id
                  ? "bg-muted text-muted-foreground"
                  : "bg-card text-card-foreground"
              }`}
              data-testid={`step-${step.id}`}
            >
              <step.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{step.title}</span>
            </div>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <div className="min-h-80">
          {renderStepContent()}
        </div>
        <div className="flex justify-between mt-6">
          <Button
            variant="outline"
            onClick={handlePrevious}
            disabled={currentStep === 1}
            data-testid="button-previous"
          >
            Previous
          </Button>
          <Button
            onClick={handleNext}
            data-testid="button-next"
            className="flex items-center gap-2"
          >
            {currentStep === steps.length ? "Analyze Patient" : "Next"}
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}