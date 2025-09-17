import { 
  Stethoscope, 
  ClipboardList, 
  TrendingUp, 
  AlertTriangle, 
  Brain, 
  FileText,
  Users,
  Settings,
  Bot 
} from "lucide-react";
import { Link, useLocation } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const menuItems = [
  {
    title: "Multi AI Diagnostic",
    url: "/multi-ai",
    icon: Bot,
    isHighlight: true,
  },
  {
    title: "Dashboard",
    url: "/",
    icon: Stethoscope,
  },
  {
    title: "Patient Intake",
    url: "/intake",
    icon: ClipboardList,
  },
  {
    title: "Diagnostics",
    url: "/diagnostics",
    icon: Brain,
  },
  {
    title: "Risk Assessment",
    url: "/risk",
    icon: TrendingUp,
  },
  {
    title: "Alerts",
    url: "/alerts",
    icon: AlertTriangle,
  },
  {
    title: "Reports",
    url: "/reports",
    icon: FileText,
  },
];

const adminItems = [
  {
    title: "Patients",
    url: "/patients",
    icon: Users,
  },
  {
    title: "Settings",
    url: "/settings",
    icon: Settings,
  },
];

export default function MedicalSidebar() {
  const [location] = useLocation();

  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-sm font-semibold text-primary">
            Clinical Tools
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton 
                    asChild
                    isActive={location === item.url}
                    size={item.isHighlight ? "lg" : "default"}
                    data-testid={`nav-${item.title.toLowerCase().replace(' ', '-')}`}
                    className={item.isHighlight ? "bg-gradient-to-r from-yellow-400 to-yellow-500 hover:from-yellow-500 hover:to-yellow-600 text-black font-semibold shadow-lg border border-yellow-300" : ""}
                  >
                    <Link href={item.url}>
                      <item.icon className={item.isHighlight ? "h-6 w-6" : "h-4 w-4"} />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-sm font-semibold text-muted-foreground">
            Administration
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {adminItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton 
                    asChild
                    isActive={location === item.url}
                    data-testid={`nav-${item.title.toLowerCase()}`}
                  >
                    <Link href={item.url}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}