import { users, type User, type SafeUser, type InsertUser } from "@shared/schema";
import { db } from "./db";
import { eq } from "drizzle-orm";
import bcrypt from "bcrypt";

// modify the interface with any CRUD methods
// you might need

export interface IStorage {
  getUser(id: string): Promise<SafeUser | undefined>;
  getUserByUsername(username: string): Promise<SafeUser | undefined>;
  getUserWithPasswordByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<SafeUser>;
  verifyPassword(plainPassword: string, hashedPassword: string): Promise<boolean>;
}

// Database storage implementation - replaces MemStorage for persistence
export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<SafeUser | undefined> {
    const [user] = await db.select({
      id: users.id,
      username: users.username
    }).from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<SafeUser | undefined> {
    const [user] = await db.select({
      id: users.id,
      username: users.username
    }).from(users).where(eq(users.username, username));
    return user || undefined;
  }

  // Internal method for authentication that includes password
  async getUserWithPasswordByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<SafeUser> {
    // Hash password before storing with increased cost factor for better security
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(insertUser.password, saltRounds);
    
    const [user] = await db
      .insert(users)
      .values({
        ...insertUser,
        password: hashedPassword
      })
      .returning();
    
    // Return user without password field for security
    const { password, ...safeUser } = user;
    return safeUser;
  }

  async verifyPassword(plainPassword: string, hashedPassword: string): Promise<boolean> {
    return bcrypt.compare(plainPassword, hashedPassword);
  }
}

export const storage = new DatabaseStorage();
