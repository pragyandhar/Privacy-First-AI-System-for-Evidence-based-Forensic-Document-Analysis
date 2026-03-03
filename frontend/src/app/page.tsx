"use client";

import { useAuth } from "@/components/AuthContext";
import ChatWindow from "@/components/ChatWindow";
import LoginPage from "@/components/LoginPage";

export default function Home() {
    const { authenticated } = useAuth();

    if (!authenticated) {
        return <LoginPage />;
    }

    return (
        <main className="flex h-screen">
            <ChatWindow />
        </main>
    );
}
