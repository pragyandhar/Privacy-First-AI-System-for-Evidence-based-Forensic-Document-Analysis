"use client";

import { useState, useEffect } from "react";
import {
    fetchTemplates,
    createTemplate,
    deleteTemplate,
    type PromptTemplate,
} from "@/lib/api";
import { FileCode2, Plus, Trash2, X, Loader2, CheckCircle, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
    open: boolean;
    onClose: () => void;
    activeTemplateId: string | null;
    onSelectTemplate: (templateId: string | null) => void;
}

export default function PromptTemplatesPanel({
    open,
    onClose,
    activeTemplateId,
    onSelectTemplate,
}: Props) {
    const [templates, setTemplates] = useState<PromptTemplate[]>([]);
    const [loading, setLoading] = useState(false);
    const [showForm, setShowForm] = useState(false);
    const [expandedId, setExpandedId] = useState<string | null>(null);

    // Form state
    const [formName, setFormName] = useState("");
    const [formDesc, setFormDesc] = useState("");
    const [formPrompt, setFormPrompt] = useState("");
    const [saving, setSaving] = useState(false);

    const loadTemplates = async () => {
        setLoading(true);
        try {
            const data = await fetchTemplates();
            setTemplates(data);
        } catch {
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (open) loadTemplates();
    }, [open]);

    const handleCreate = async () => {
        if (!formName.trim() || !formPrompt.trim()) return;
        setSaving(true);
        try {
            await createTemplate({
                id: "",
                name: formName.trim(),
                description: formDesc.trim(),
                prompt: formPrompt.trim(),
            });
            setFormName("");
            setFormDesc("");
            setFormPrompt("");
            setShowForm(false);
            await loadTemplates();
        } catch {
        } finally {
            setSaving(false);
        }
    };

    const handleDelete = async (id: string) => {
        try {
            await deleteTemplate(id);
            if (activeTemplateId === id) onSelectTemplate(null);
            await loadTemplates();
        } catch {
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-2xl max-h-[80vh] bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <FileCode2 size={18} className="text-purple-400" />
                        <h2 className="text-sm font-semibold">Prompt Templates</h2>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowForm((v) => !v)}
                            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium transition-colors"
                        >
                            <Plus size={14} />
                            Add Custom
                        </button>
                        <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
                    {/* Active template indicator */}
                    {activeTemplateId && (
                        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-900/20 text-purple-300 text-xs">
                            <CheckCircle size={14} />
                            <span>
                                Active template: <strong>{templates.find((t) => t.id === activeTemplateId)?.name ?? activeTemplateId}</strong>
                            </span>
                            <button
                                onClick={() => onSelectTemplate(null)}
                                className="ml-auto text-[10px] px-2 py-0.5 rounded bg-purple-800/50 hover:bg-purple-700/50"
                            >
                                Clear
                            </button>
                        </div>
                    )}

                    {/* New template form */}
                    {showForm && (
                        <div className="border border-purple-800/30 rounded-xl p-4 bg-purple-900/10 space-y-3">
                            <p className="text-xs font-semibold text-purple-300">Create Custom Template</p>
                            <input
                                value={formName}
                                onChange={(e) => setFormName(e.target.value)}
                                placeholder="Template name"
                                className="w-full rounded-lg bg-[var(--background)] border border-[var(--border)] px-3 py-2 text-sm placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-purple-500"
                            />
                            <input
                                value={formDesc}
                                onChange={(e) => setFormDesc(e.target.value)}
                                placeholder="Short description"
                                className="w-full rounded-lg bg-[var(--background)] border border-[var(--border)] px-3 py-2 text-sm placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-purple-500"
                            />
                            <textarea
                                value={formPrompt}
                                onChange={(e) => setFormPrompt(e.target.value)}
                                placeholder="Prompt template (use {context} and {input} as placeholders)"
                                rows={5}
                                className="w-full rounded-lg bg-[var(--background)] border border-[var(--border)] px-3 py-2 text-sm placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-purple-500 resize-none"
                            />
                            <div className="flex gap-2">
                                <button
                                    onClick={handleCreate}
                                    disabled={saving || !formName.trim() || !formPrompt.trim()}
                                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white text-sm font-medium transition-colors"
                                >
                                    {saving ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                                    Save
                                </button>
                                <button
                                    onClick={() => setShowForm(false)}
                                    className="px-4 py-2 rounded-lg bg-[var(--card-hover)] text-sm text-[var(--muted)]"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}

                    {loading ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 size={24} className="animate-spin text-purple-400" />
                        </div>
                    ) : (
                        templates.map((t) => {
                            const isActive = activeTemplateId === t.id;
                            const isExpanded = expandedId === t.id;
                            return (
                                <div
                                    key={t.id}
                                    className={`border rounded-xl overflow-hidden transition-colors ${isActive
                                        ? "border-purple-500/50 bg-purple-900/15"
                                        : "border-[var(--border)] bg-[var(--background)]"
                                        }`}
                                >
                                    <div className="flex items-center gap-2 px-4 py-3">
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <span className="text-sm font-medium truncate">{t.name}</span>
                                                {t.is_default && (
                                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-600/20 text-brand-300">
                                                        default
                                                    </span>
                                                )}
                                                {isActive && (
                                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-600/30 text-purple-300">
                                                        active
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-xs text-[var(--muted)] truncate">{t.description}</p>
                                        </div>
                                        <button
                                            onClick={() => setExpandedId(isExpanded ? null : t.id)}
                                            className="p-1 rounded hover:bg-[var(--card-hover)]"
                                        >
                                            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>
                                        <button
                                            onClick={() => onSelectTemplate(isActive ? null : t.id)}
                                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${isActive
                                                ? "bg-purple-600 text-white"
                                                : "bg-[var(--card-hover)] text-[var(--muted)] hover:text-white"
                                                }`}
                                        >
                                            {isActive ? "Active" : "Use"}
                                        </button>
                                        {!t.is_default && (
                                            <button
                                                onClick={() => handleDelete(t.id)}
                                                className="p-1.5 rounded-lg text-red-400 hover:bg-red-900/20"
                                                title="Delete template"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        )}
                                    </div>
                                    {isExpanded && (
                                        <div className="px-4 pb-3 border-t border-[var(--border)]">
                                            <pre className="text-[11px] text-[var(--muted)] whitespace-pre-wrap mt-2 leading-relaxed">
                                                {t.prompt}
                                            </pre>
                                        </div>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </div>
    );
}
