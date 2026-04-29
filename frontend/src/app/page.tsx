"use client";

import { useState, useEffect } from "react";
import ChatWindow from "@/components/ChatWindow";
import AuthScreen from "@/components/AuthScreen";
import { getToken, getStoredUser, logout, type AuthUser } from "@/lib/api";

export default function Home() {
    const [user, setUser] = useState<AuthUser | null>(null);
    const [checking, setChecking] = useState(true);

    useEffect(() => {
        // Check for existing token on mount
        const token = getToken();
        const stored = getStoredUser();
        if (token && stored) {
            setUser(stored);
        }
        setChecking(false);

        // Listen for 401 events (token expired)
        const handleUnauth = () => {
            setUser(null);
        };
        window.addEventListener("neuravault:unauthorized", handleUnauth);
        return () => window.removeEventListener("neuravault:unauthorized", handleUnauth);
    }, []);

    const handleLogout = () => {
        logout();
        setUser(null);
    };

    if (checking) {
        return (
            <main className="flex h-screen items-center justify-center">
                <div className="animate-pulse text-[var(--muted)]">Loading…</div>
            </main>
        );
    }

    if (!user) {
        return <AuthScreen onAuthenticated={setUser} />;
    }

    return (
        <main className="flex h-screen">
            <ChatWindow user={user} onLogout={handleLogout} />
        </main>
    );
}
