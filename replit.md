# AI-Powered Diagnostic & Triage Assistant

## Overview

This is a modern web application for AI-powered medical diagnostic and triage support designed to assist clinicians. The system provides intelligent data synthesis, ranked diagnosis lists with confidence levels, risk stratification, and red-flag alerts for urgent conditions. Built with a focus on explainability and trust, the application features a clean medical dashboard interface with comprehensive patient intake forms, diagnostic analysis, and detailed reporting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **UI Library**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with custom medical-focused design system
- **State Management**: TanStack Query (React Query) for server state management
- **Routing**: Wouter for client-side routing
- **Theme System**: Dark/light mode support with custom CSS variables

### Design System
- **Color Palette**: Medical-focused with clinical blue primary, alert system colors (critical red, warning amber, success green)
- **Typography**: Inter for primary text, JetBrains Mono for medical data precision
- **Layout**: Material Design principles adapted for medical workflows
- **Component Library**: Comprehensive set of medical-specific components including diagnostic cards, progress indicators, alert banners, and risk assessment displays

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **File Uploads**: Multer for handling medical document uploads (PDF, images, text files)
- **Security**: Helmet.js for security headers with environment-specific CSP
- **API Structure**: RESTful endpoints for medical analysis and file processing

### Data Management
- **Database**: PostgreSQL using Neon serverless
- **ORM**: Drizzle ORM for type-safe database operations
- **Schema**: User management with bcrypt password hashing
- **File Storage**: Local file system for uploaded medical documents

### AI Integration
- **Primary AI**: Advanced VibeyBot engine for medical document analysis
- **Analysis Pipeline**: Four-stage medical analysis (intake, analysis, triage, explanation)
- **Fallback Handling**: Graceful degradation when external services are unavailable
- **Multi-AI Interface**: Support for processing multiple document types with animated bot workflows

### Authentication & Security
- **Password Security**: bcrypt with 12 salt rounds
- **Session Management**: Cookie-based sessions
- **Data Sanitization**: Sensitive field filtering in logs
- **File Upload Security**: MIME type validation and size limits
- **CSP**: Environment-specific Content Security Policy

### Key Features
- **Patient Intake**: Multi-step forms for comprehensive patient data collection
- **Diagnostic Analysis**: AI-powered diagnosis ranking with confidence levels and evidence mapping
- **Risk Assessment**: Automated risk stratification with visual indicators
- **Alert System**: Red-flag detection for urgent medical conditions
- **Explainability**: Transparent AI decision-making with reasoning steps
- **Medical Dashboard**: Real-time statistics, trends, and recent activity monitoring
- **Document Processing**: Support for various medical document formats

## External Dependencies

### Core Framework Dependencies
- **React Ecosystem**: React 18+ with TypeScript, Vite build system
- **UI Components**: Radix UI primitives for accessible component foundation
- **Styling**: Tailwind CSS with PostCSS processing

### AI & Machine Learning
- **VibeyBot Engine**: Primary AI service for medical analysis with advanced fallback capabilities
- **Smart Processing**: Intelligent document analysis with multiple processing stages

### Database & Backend
- **PostgreSQL**: Neon serverless database (@neondatabase/serverless)
- **Drizzle ORM**: Type-safe database operations with migrations
- **Express.js**: Web framework with security middleware

### Security & Authentication
- **bcrypt**: Password hashing for user authentication
- **Helmet**: Security headers and CSP management
- **Multer**: Secure file upload handling

### Development & Build Tools
- **Vite**: Fast build tool with development server
- **ESBuild**: JavaScript bundling for production
- **TypeScript**: Static type checking
- **Replit Integration**: Development environment support

### Utility Libraries
- **TanStack Query**: Server state management and caching
- **React Hook Form**: Form validation and management
- **Wouter**: Lightweight client-side routing
- **date-fns**: Date manipulation utilities
- **Framer Motion**: Animation library for UI interactions

### Visualization
- **Recharts**: Medical data visualization and charting
- **Lucide React**: Medical and general iconography

The application is designed to be deployed on Replit with PostgreSQL database provisioning and environment variable configuration for AI services.