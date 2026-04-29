"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
    queryRAG,
    fetchStarters,
    fetchHealth,
    uploadFiles,
    ingestDocuments,
    queryExplainableStream,
    type QueryResponse,
    type StarterQuestion,
    type SourceInfo,
    type ChatMessageExport,
} from "@/lib/api";
import MessageBubble, { type ChatMessage } from "@/components/MessageBubble";
import StarterQuestions from "@/components/StarterQuestions";
import Sidebar from "@/components/Sidebar";
import TimelinePanel from "@/components/TimelinePanel";
import AnomalyPanel from "@/components/AnomalyPanel";
import ExportMenu from "@/components/ExportMenu";
import EvidenceGraphPanel from "@/components/EvidenceGraphPanel";
import PromptTemplatesPanel from "@/components/PromptTemplatesPanel";
import AnalyticsPanel from "@/components/AnalyticsPanel";
import { Send, Upload, Menu, Download, Brain, LogOut, Shield } from "lucide-react";
import { type AuthUser } from "@/lib/api";

interface ChatWindowProps {
    user: AuthUser;
    onLogout: () => void;
}

export default function ChatWindow({ user, onLogout }: ChatWindowProps) {
    /* ---- state ------------------------------------------------------------ */
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [starters, setStarters] = useState<StarterQuestion[]>([]);
    const [engineReady, setEngineReady] = useState<boolean | null>(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);

    // Feature panel states
    const [timelineOpen, setTimelineOpen] = useState(false);
    const [anomalyOpen, setAnomalyOpen] = useState(false);
    const [exportOpen, setExportOpen] = useState(false);
    const [evidenceOpen, setEvidenceOpen] = useState(false);
    const [templatesOpen, setTemplatesOpen] = useState(false);
    const [analyticsOpen, setAnalyticsOpen] = useState(false);

    // Explainable AI / prompt template state
    const [explainMode, setExplainMode] = useState(false);
    const [activeTemplateId, setActiveTemplateId] = useState<string | null>(null);

    const bottomRef = useRef<HTMLDivElement>(null);
    const fileRef = useRef<HTMLInputElement>(null);

    /* ---- effects ---------------------------------------------------------- */
    useEffect(() => {
        fetchStarters()
            .then(setStarters)
            .catch(() => { });
        fetchHealth()
            .then((h) => setEngineReady(h.engine_ready))
            .catch(() => setEngineReady(false));
    }, []);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    /* ---- handlers --------------------------------------------------------- */
    const sendMessage = useCallback(
        async (text?: string) => {
            const question = (text ?? input).trim();
            if (!question || loading) return;

            setInput("");
            const userMsg: ChatMessage = { role: "user", content: question };
            setMessages((prev) => [...prev, userMsg]);
            setLoading(true);

            try {
                if (explainMode) {
                    // Streaming explainable AI mode
                    let streamedContent = "";
                    let sources: SourceInfo[] = [];

                    // Add a placeholder message that we'll update
                    const placeholderIdx = messages.length + 1; // +1 for user msg
                    setMessages((prev) => [
                        ...prev,
                        { role: "assistant", content: "**REASONING STEPS:**\n", sources: [] },
                    ]);

                    await queryExplainableStream(
                        question,
                        activeTemplateId ?? undefined,
                        (token) => {
                            streamedContent += token;
                            setMessages((prev) => {
                                const updated = [...prev];
                                const lastIdx = updated.length - 1;
                                if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
                                    updated[lastIdx] = {
                                        ...updated[lastIdx],
                                        content: streamedContent,
                                    };
                                }
                                return updated;
                            });
                        },
                        (srcs) => {
                            sources = srcs;
                        },
                        () => {
                            // Done - update sources
                            setMessages((prev) => {
                                const updated = [...prev];
                                const lastIdx = updated.length - 1;
                                if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
                                    updated[lastIdx] = {
                                        ...updated[lastIdx],
                                        sources: sources,
                                    };
                                }
                                return updated;
                            });
                        },
                        (error) => {
                            setMessages((prev) => {
                                const updated = [...prev];
                                const lastIdx = updated.length - 1;
                                if (lastIdx >= 0) {
                                    updated[lastIdx] = {
                                        role: "assistant",
                                        content: `**Error:** ${error}`,
                                    };
                                }
                                return updated;
                            });
                        }
                    );
                } else {
                    // Standard query mode
                    const res: QueryResponse = await queryRAG(question);
                    const assistantMsg: ChatMessage = {
                        role: "assistant",
                        content: res.answer,
                        sources: res.sources,
                    };
                    setMessages((prev) => [...prev, assistantMsg]);
                }
            } catch (err: any) {
                const errMsg: ChatMessage = {
                    role: "assistant",
                    content: `**Error:** ${err.message ?? "Something went wrong."}`,
                };
                setMessages((prev) => [...prev, errMsg]);
            } finally {
                setLoading(false);
            }
        },
        [input, loading, explainMode, activeTemplateId]
    );

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const sysMsg: ChatMessage = {
            role: "assistant",
            content: `Uploading **${files.length}** file(s)…`,
        };
        setMessages((prev) => [...prev, sysMsg]);
        setLoading(true);

        try {
            const uploadRes = await uploadFiles(files);
            const ingestRes = await ingestDocuments();
            const doneMsg: ChatMessage = {
                role: "assistant",
                content: `✅ Upload complete! Processed **${ingestRes.documents_processed}** document(s) into **${ingestRes.chunks_created}** chunks.\n\nYou can now ask questions about the new documents.`,
            };
            setMessages((prev) => [...prev, doneMsg]);
            setEngineReady(true);
        } catch (err: any) {
            const errMsg: ChatMessage = {
                role: "assistant",
                content: `**Upload failed:** ${err.message}`,
            };
            setMessages((prev) => [...prev, errMsg]);
        } finally {
            setLoading(false);
            if (fileRef.current) fileRef.current.value = "";
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    // Prepare messages for export
    const exportMessages: ChatMessageExport[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
    }));

    /* ---- render ----------------------------------------------------------- */
    return (
        <div className="flex w-full h-full">
            {/* Sidebar */}
            <Sidebar
                open={sidebarOpen}
                onClose={() => setSidebarOpen(false)}
                engineReady={engineReady}
                onEngineReady={() => setEngineReady(true)}
                onOpenTimeline={() => { setTimelineOpen(true); setSidebarOpen(false); }}
                onOpenAnomalies={() => { setAnomalyOpen(true); setSidebarOpen(false); }}
                onOpenEvidenceGraph={() => { setEvidenceOpen(true); setSidebarOpen(false); }}
                onOpenTemplates={() => { setTemplatesOpen(true); setSidebarOpen(false); }}
                onOpenAnalytics={() => { setAnalyticsOpen(true); setSidebarOpen(false); }}
            />

            <div className="flex flex-col flex-1 h-full">
                {/* Header */}
                <header className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)] bg-[var(--card)]">
                    <button
                        onClick={() => setSidebarOpen((v) => !v)}
                        className="p-1.5 rounded-lg hover:bg-[var(--card-hover)] transition-colors lg:hidden"
                    >
                        <Menu size={20} />
                    </button>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold text-sm">
                            NV
                        </div>
                        <div>
                            <h1 className="text-sm font-semibold leading-tight">NeuraVault</h1>
                            <p className="text-xs text-[var(--muted)]">
                                Privacy-first forensic RAG
                            </p>
                        </div>
                    </div>

                    {/* Header action buttons */}
                    <div className="ml-auto flex items-center gap-2">
                        {/* User badge */}
                        <span className="hidden sm:flex items-center gap-1.5 text-xs text-[var(--muted)] px-2 py-1 rounded-lg bg-[var(--background)]">
                            <Shield size={12} className="text-brand-400" />
                            {user.username}
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-600/20 text-brand-300">
                                {user.role}
                            </span>
                        </span>
                        {/* Explain mode toggle */}
                        <button
                            onClick={() => setExplainMode((v) => !v)}
                            title={explainMode ? "Explainable AI ON (streaming)" : "Enable Explainable AI"}
                            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${explainMode
                                ? "bg-purple-600/30 text-purple-300 border border-purple-500/40"
                                : "bg-[var(--card-hover)] text-[var(--muted)] hover:text-white"
                                }`}
                        >
                            <Brain size={14} />
                            {explainMode ? "Explain ON" : "Explain"}
                        </button>

                        {/* Export button */}
                        <button
                            onClick={() => setExportOpen(true)}
                            disabled={messages.length === 0}
                            title="Export chat"
                            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-[var(--card-hover)] text-[var(--muted)] hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            <Download size={14} />
                            Export
                        </button>

                        {/* Engine status */}
                        {engineReady !== null && (
                            <span
                                className={`text-xs px-2 py-0.5 rounded-full ${engineReady
                                    ? "bg-green-900/40 text-green-400"
                                    : "bg-yellow-900/40 text-yellow-400"
                                    }`}
                            >
                                {engineReady ? "Engine Ready" : "Engine Not Ready"}
                            </span>
                        )}

                        {/* Logout button */}
                        <button
                            onClick={onLogout}
                            title="Sign out"
                            className="p-1.5 rounded-lg hover:bg-red-600/20 text-[var(--muted)] hover:text-red-400 transition-colors"
                        >
                            <LogOut size={16} />
                        </button>
                    </div>
                </header>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
                    {messages.length === 0 && !loading && (
                        <div className="flex flex-col items-center justify-center h-full gap-6">
                            <div className="text-center space-y-2">
                                <div className="w-16 h-16 mx-auto rounded-2xl bg-brand-600/20 flex items-center justify-center">
                                    <span className="text-3xl font-bold text-brand-400">NV</span>
                                </div>
                                <h2 className="text-xl font-semibold">
                                    Welcome to NeuraVault
                                </h2>
                                <p className="text-sm text-[var(--muted)] max-w-md">
                                    Ask any question about the documents in your vault. All
                                    processing happens locally — your data never leaves this
                                    machine.
                                </p>
                            </div>
                            {starters.length > 0 && (
                                <StarterQuestions
                                    starters={starters}
                                    onSelect={(msg) => sendMessage(msg)}
                                />
                            )}
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <MessageBubble key={i} message={msg} />
                    ))}

                    {loading && (
                        <div className="flex items-start gap-3 max-w-3xl mx-auto">
                            <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center text-white text-xs font-bold shrink-0">
                                NV
                            </div>
                            <div className="px-4 py-3 rounded-2xl bg-[var(--card)] text-sm dot-animation">
                                <span>●</span> <span>●</span> <span>●</span>
                            </div>
                        </div>
                    )}

                    <div ref={bottomRef} />
                </div>

                {/* Input bar */}
                <div className="border-t border-[var(--border)] bg-[var(--card)] px-4 py-3">
                    {/* Active template indicator */}
                    {activeTemplateId && (
                        <div className="flex items-center gap-2 mb-2 max-w-3xl mx-auto text-xs text-purple-300">
                            <span className="w-2 h-2 rounded-full bg-purple-500" />
                            Using custom prompt template
                            <button
                                onClick={() => setActiveTemplateId(null)}
                                className="ml-auto text-[10px] px-2 py-0.5 rounded bg-purple-800/50 hover:bg-purple-700/50"
                            >
                                Clear
                            </button>
                        </div>
                    )}
                    <form
                        onSubmit={(e) => {
                            e.preventDefault();
                            sendMessage();
                        }}
                        className="flex items-end gap-2 max-w-3xl mx-auto"
                    >
                        {/* hidden file input */}
                        <input
                            ref={fileRef}
                            type="file"
                            accept=".pdf"
                            multiple
                            className="hidden"
                            onChange={handleUpload}
                        />
                        <button
                            type="button"
                            onClick={() => fileRef.current?.click()}
                            title="Upload PDF"
                            className="p-2.5 rounded-xl hover:bg-[var(--card-hover)] transition-colors text-[var(--muted)] hover:text-white"
                        >
                            <Upload size={18} />
                        </button>

                        <textarea
                            rows={1}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={explainMode ? "Ask with reasoning steps…" : "Ask a question about your documents…"}
                            className="flex-1 resize-none rounded-xl bg-[var(--background)] border border-[var(--border)] px-4 py-2.5 text-sm placeholder:text-[var(--muted)] focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                            style={{ maxHeight: 160 }}
                        />

                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="p-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-white"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                    <p className="text-center text-[10px] text-[var(--muted)] mt-2">
                        100 % offline · All data stays on your machine
                        {explainMode && " · Explainable AI enabled"}
                    </p>
                </div>
            </div>

            {/* Feature Panels */}
            <TimelinePanel open={timelineOpen} onClose={() => setTimelineOpen(false)} />
            <AnomalyPanel open={anomalyOpen} onClose={() => setAnomalyOpen(false)} />
            <ExportMenu open={exportOpen} onClose={() => setExportOpen(false)} messages={exportMessages} />
            <EvidenceGraphPanel open={evidenceOpen} onClose={() => setEvidenceOpen(false)} />
            <PromptTemplatesPanel
                open={templatesOpen}
                onClose={() => setTemplatesOpen(false)}
                activeTemplateId={activeTemplateId}
                onSelectTemplate={setActiveTemplateId}
            />
            <AnalyticsPanel open={analyticsOpen} onClose={() => setAnalyticsOpen(false)} />
        </div>
    );
}
