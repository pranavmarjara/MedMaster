import { SidebarProvider } from "@/components/ui/sidebar";
import MedicalSidebar from '../MedicalSidebar';

export default function MedicalSidebarExample() {
  const style = {
    "--sidebar-width": "20rem",
    "--sidebar-width-icon": "4rem",
  };

  return (
    <SidebarProvider style={style as React.CSSProperties}>
      <div className="flex h-96 w-full">
        <MedicalSidebar />
      </div>
    </SidebarProvider>
  );
}