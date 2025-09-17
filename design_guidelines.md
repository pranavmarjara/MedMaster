# VibeyBot Diagnostic & Triage Assistant - Design Guidelines

## Design Approach
**System-Based Approach** using **Material Design** principles adapted for medical applications, emphasizing clear information hierarchy, trust-building elements, and clinical workflow optimization.

## Core Design Elements

### A. Color Palette
**Primary Colors:**
- Clinical Blue: 210 85% 45% (trust, professionalism)
- Medical White: 0 0% 98% (cleanliness, clarity)
- Dark Mode Base: 220 15% 12%

**Alert System Colors:**
- Critical Red: 0 75% 55% (urgent conditions)
- Warning Amber: 35 85% 55% (moderate risk)
- Success Green: 120 65% 45% (low risk/normal)
- Info Blue: 210 85% 65% (informational)

### B. Typography
- **Primary:** Inter (via Google Fonts)
- **Medical Data:** JetBrains Mono (monospace for clinical precision)
- Headers: 600-700 weight, body text: 400-500 weight

### C. Layout System
**Tailwind Spacing Units:** 2, 4, 6, 8, 12, 16
- Tight spacing (p-2, m-2) for data-dense areas
- Generous spacing (p-8, m-12) for workflow sections
- Consistent 16-unit vertical rhythm for major sections

### D. Component Library

**Core Components:**
- **Progress Indicators:** Multi-step forms with animated progress bars
- **Diagnostic Cards:** Color-coded risk levels with confidence percentages
- **Alert Banners:** Prominent red-flag warnings with animated pulse effects
- **Data Visualization:** Clean charts with medical-appropriate color coding
- **Input Forms:** Floating labels with real-time validation feedback

**Navigation:**
- Fixed sidebar with medical iconography
- Breadcrumb navigation for complex workflows
- Quick-access floating action button for emergency protocols

**Data Displays:**
- Structured medical reports with expandable sections
- Patient timeline with interactive milestones
- Confidence meters with animated fill states

**Overlays:**
- Modal dialogs for detailed diagnostic explanations
- Toast notifications for system status updates
- Loading overlays with medical-themed animations (pulse, heartbeat rhythm)

### E. Animations
**Minimal, Purposeful Animations:**
- Loading states: Subtle pulse effect for VibeyBot processing
- Form validation: Smooth color transitions for feedback
- Alert animations: Gentle glow for critical alerts
- Page transitions: Slide effects between workflow steps
- Data reveals: Fade-in animations for diagnostic results

## Medical-Specific Design Considerations

**Trust Elements:**
- Professional color scheme avoiding playful elements
- Clear confidence indicators for VibeyBot recommendations
- Prominent disclaimers and human oversight reminders
- Evidence-based reasoning displays

**Workflow Optimization:**
- Single-column layouts for form completion
- Fixed action buttons for critical functions
- Clear visual hierarchy for triage priorities
- Tablet-optimized touch targets (minimum 44px)

**Data Presentation:**
- High contrast ratios for clinical readability
- Structured layouts mimicking medical forms
- Color-coded risk stratification
- Clear separation between patient data and VibeyBot analysis

## Images
No large hero images required. Focus on:
- Medical iconography (stethoscope, charts, diagnostic symbols)
- Data visualization graphics
- Progress indicators and status icons
- Professional, clinical aesthetic throughout