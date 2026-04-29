"use client";

import { useState } from "react";
import { exportChat, type ChatMessageExport } from "@/lib/api";
import { Download, FileText, File, X, Loader2 } from "lucide-react";

interface Props {
    open: boolean;
    onClose: () => void;
    messages: ChatMessageExport[];
}

export default function ExportMenu({ open, onClose, messages }: Props) {
    const [exporting, setExporting] = useState(false);
    const [format, setFormat] = useState<"pdf" | "docx">("pdf");

    const handleExport = async () => {
        if (messages.length === 0) return;
        setExporting(true);
        try {
            const blob = await exportChat(messages, format);
            // Trigger download
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `neuravault_chat.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            onClose();
        } catch (err: any) {
            alert("Export failed: " + (err.message ?? "Unknown error"));
        } finally {
            setExporting(false);
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-sm bg-[var(--card)] border border-[var(--border)] rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <Download size={18} className="text-brand-400" />
                        <h2 className="text-sm font-semibold">Export Chat</h2>
                    </div>
                    <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="px-5 py-5 space-y-4">
                    <p className="text-sm text-[var(--muted)]">
                        Export {messages.length} message(s) with sources and citations.
                    </p>

                    {/* Format options */}
                    <div className="space-y-2">
                        <p className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Format</p>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setFormat("pdf")}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl border text-sm transition-colors ${format === "pdf"
                                    ? "border-red-500/50 bg-red-900/15 text-red-300"
                                    : "border-[var(--border)] bg-[var(--background)] text-[var(--muted)] hover:border-red-500/30"
                                    }`}
                            >
                                <FileText size={18} />
                                PDF
                            </button>
                            <button
                                onClick={() => setFormat("docx")}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl border text-sm transition-colors ${format === "docx"
                                    ? "border-blue-500/50 bg-blue-900/15 text-blue-300"
                                    : "border-[var(--border)] bg-[var(--background)] text-[var(--muted)] hover:border-blue-500/30"
                                    }`}
                            >
                                <File size={18} />
                                DOCX
                            </button>
                        </div>
                    </div>

                    {/* Export button */}
                    <button
                        onClick={handleExport}
                        disabled={exporting || messages.length === 0}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-white text-sm font-medium"
                    >
                        {exporting ? (
                            <Loader2 size={16} className="animate-spin" />
                        ) : (
                            <Download size={16} />
                        )}
                        {exporting ? "Exporting…" : `Export as ${format.toUpperCase()}`}
                    </button>

                    {messages.length === 0 && (
                        <p className="text-xs text-yellow-400 text-center">
                            No messages to export. Start a conversation first.
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}
