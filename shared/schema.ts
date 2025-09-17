import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const medicalAnalyses = pgTable("medical_analyses", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fileName: text("file_name").notNull(),
  fileType: text("file_type").notNull(),
  intake: text("intake").notNull(),
  analysis: text("analysis").notNull(),
  triage: text("triage").notNull(),
  explanation: text("explanation").notNull(),
  processingTimeMs: integer("processing_time_ms"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  userId: varchar("user_id"), // Optional for future user association
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertMedicalAnalysisSchema = createInsertSchema(medicalAnalyses).pick({
  fileName: true,
  fileType: true,
  intake: true,
  analysis: true,
  triage: true,
  explanation: true,
  processingTimeMs: true,
  userId: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type SafeUser = Omit<User, 'password'>;
export type MedicalAnalysis = typeof medicalAnalyses.$inferSelect;
export type InsertMedicalAnalysis = z.infer<typeof insertMedicalAnalysisSchema>;
