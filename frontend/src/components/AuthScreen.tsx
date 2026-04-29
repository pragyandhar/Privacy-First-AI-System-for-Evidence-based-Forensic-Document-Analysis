"use client";

import { useState } from "react";
import { login, register, type LoginResponse } from "@/lib/api";
import { Lock, User, Eye, EyeOff, ShieldCheck, AlertCircle } from "lucide-react";

interface Props {
    onAuthenticated: (user: { username: string; role: string }) => void;
}

export default function AuthScreen({ onAuthenticated }: Props) {
    const [mode, setMode] = useState<"login" | "register">("login");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [role, setRole] = useState("analyst");
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const toErrorMessage = (value: unknown): string => {
        if (typeof value === "string") return value;
        if (value && typeof value === "object") {
            if ("msg" in value) return String((value as { msg?: unknown }).msg ?? "Authentication failed.");
            try {
                return JSON.stringify(value);
            } catch {
                return "Authentication failed.";
            }
        }
        return "Authentication failed.";
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);

        try {
            let res: LoginResponse;
            if (mode === "login") {
                res = await login(username, password);
            } else {
                res = await register(username, password, role);
            }
            onAuthenticated({ username: res.username, role: res.role });
        } catch (err: unknown) {
            let msg = err instanceof Error ? err.message : "Authentication failed.";
            try {
                const parsed = JSON.parse(msg);
                const detail = parsed?.detail;
                if (Array.isArray(detail)) {
                    const messages = detail.map(toErrorMessage).filter(Boolean);
                    msg = messages.length > 0 ? messages.join(" | ") : msg;
                } else if (detail) {
                    msg = toErrorMessage(detail);
                }
            } catch { }
            setError(toErrorMessage(msg));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-[var(--background)] p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-brand-600 mb-4">
                        <ShieldCheck size={32} className="text-white" />
                    </div>
                    <h1 className="text-2xl font-bold">NeuraVault</h1>
                    <p className="text-sm text-[var(--muted)] mt-1">
                        Privacy-First Forensic Analysis System
                    </p>
                </div>

                {/* Card */}
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-lg">
                    {/* Tabs */}
                    <div className="flex mb-6 bg-[var(--background)] rounded-lg p-1">
                        <button
                            onClick={() => { setMode("login"); setError(null); }}
                            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${mode === "login"
                                ? "bg-brand-600 text-white"
                                : "text-[var(--muted)] hover:text-[var(--foreground)]"
                                }`}
                        >
                            Sign In
                        </button>
                        <button
                            onClick={() => { setMode("register"); setError(null); }}
                            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${mode === "register"
                                ? "bg-brand-600 text-white"
                                : "text-[var(--muted)] hover:text-[var(--foreground)]"
                                }`}
                        >
                            Register
                        </button>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-start gap-2">
                            <AlertCircle size={16} className="text-red-400 shrink-0 mt-0.5" />
                            <p className="text-sm text-red-400">{error}</p>
                        </div>
                    )}

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* Username */}
                        <div>
                            <label className="block text-xs font-medium text-[var(--muted)] mb-1.5">
                                Username
                            </label>
                            <div className="relative">
                                <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                                <input
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    placeholder="Enter username"
                                    required
                                    minLength={3}
                                    maxLength={50}
                                    className="w-full pl-10 pr-3 py-2.5 bg-[var(--background)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                                />
                            </div>
                        </div>

                        {/* Password */}
                        <div>
                            <label className="block text-xs font-medium text-[var(--muted)] mb-1.5">
                                Password
                            </label>
                            <div className="relative">
                                <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder={mode === "register" ? "Min 8 chars, upper+lower+digit+special" : "Enter password"}
                                    required
                                    minLength={8}
                                    className="w-full pl-10 pr-10 py-2.5 bg-[var(--background)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--muted)] hover:text-[var(--foreground)]"
                                >
                                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                            {mode === "register" && (
                                <p className="text-xs text-[var(--muted)] mt-1">
                                    Must contain uppercase, lowercase, digit, and special character.
                                </p>
                            )}
                        </div>

                        {/* Role selector (register only) */}
                        {mode === "register" && (
                            <div>
                                <label className="block text-xs font-medium text-[var(--muted)] mb-1.5">
                                    Role
                                </label>
                                <select
                                    value={role}
                                    onChange={(e) => setRole(e.target.value)}
                                    className="w-full px-3 py-2.5 bg-[var(--background)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500"
                                >
                                    <option value="viewer">Viewer — Read-only access</option>
                                    <option value="analyst">Analyst — Upload & query</option>
                                    <option value="admin">Admin — Full access</option>
                                </select>
                            </div>
                        )}

                        {/* Submit */}
                        <button
                            type="submit"
                            disabled={loading || !username || !password}
                            className="w-full py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg text-sm transition-colors"
                        >
                            {loading
                                ? "Please wait…"
                                : mode === "login"
                                    ? "Sign In"
                                    : "Create Account"}
                        </button>
                    </form>
                </div>

                {/* Security badges */}
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {["Encrypted", "Offline", "bcrypt", "JWT Auth"].map((badge) => (
                        <span
                            key={badge}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-brand-600/10 text-brand-400 text-xs"
                        >
                            <ShieldCheck size={10} />
                            {badge}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}
