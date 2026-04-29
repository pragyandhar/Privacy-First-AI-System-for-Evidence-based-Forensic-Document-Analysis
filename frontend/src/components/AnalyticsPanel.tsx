"use client";

import { useState, useEffect } from "react";
import {
    fetchAnalyticsSummary,
    type AnalyticsSummary,
    type DailyCount,
    type DailyQuality,
} from "@/lib/api";
import { BarChart3, Loader2, X, RefreshCw } from "lucide-react";

interface Props {
    open: boolean;
    onClose: () => void;
}

function MiniBarList({ items, max }: { items: DailyCount[]; max: number }) {
    if (items.length === 0) return null;
    return (
        <div className="space-y-1">
            {items.map((row) => (
                <div key={row.date} className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--muted)] w-20 shrink-0">{row.date}</span>
                    <div className="flex-1 h-2 bg-[var(--border)] rounded-full overflow-hidden">
                        <div
                            className="h-full bg-brand-500"
                            style={{ width: `${Math.min(100, (row.count / max) * 100)}%` }}
                        />
                    </div>
                    <span className="text-[10px] text-[var(--muted)] w-8 text-right">{row.count}</span>
                </div>
            ))}
        </div>
    );
}

function MiniRateList({ items }: { items: DailyQuality[] }) {
    if (items.length === 0) return null;
    return (
        <div className="space-y-1">
            {items.map((row) => (
                <div key={row.date} className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--muted)] w-20 shrink-0">{row.date}</span>
                    <div className="flex-1 h-2 bg-[var(--border)] rounded-full overflow-hidden">
                        <div
                            className="h-full bg-green-500"
                            style={{ width: `${Math.min(100, row.citation_rate)}%` }}
                        />
                    </div>
                    <span className="text-[10px] text-[var(--muted)] w-16 text-right">
                        {row.citation_rate}%
                    </span>
                </div>
            ))}
        </div>
    );
}

export default function AnalyticsPanel({ open, onClose }: Props) {
    const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
    const [loading, setLoading] = useState(false);
    const [days, setDays] = useState(30);
    const [error, setError] = useState<string | null>(null);

    const loadSummary = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAnalyticsSummary(days);
            setSummary(data);
        } catch (err: any) {
            setError(err.message ?? "Failed to load analytics");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (open) {
            loadSummary();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, days]);

    if (!open) return null;

    const usage = summary?.usage;
    const rag = summary?.rag_quality;

    const maxDailyQueries = usage?.daily_queries?.reduce((m, d) => Math.max(m, d.count), 1) ?? 1;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-4xl max-h-[90vh] bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <BarChart3 size={18} className="text-brand-400" />
                        <h2 className="text-sm font-semibold">Analytics Dashboard</h2>
                    </div>
                    <div className="flex items-center gap-2">
                        <select
                            value={days}
                            onChange={(e) => setDays(Number(e.target.value))}
                            className="text-xs bg-[var(--background)] border border-[var(--border)] rounded-lg px-2 py-1"
                        >
                            {[7, 14, 30, 60, 90].map((d) => (
                                <option key={d} value={d}>{d} days</option>
                            ))}
                        </select>
                        <button
                            onClick={loadSummary}
                            className="p-1.5 rounded-lg hover:bg-[var(--card-hover)]"
                            title="Refresh"
                        >
                            <RefreshCw size={14} />
                        </button>
                        <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
                    {loading && (
                        <div className="flex flex-col items-center gap-3 py-10">
                            <Loader2 size={28} className="animate-spin text-brand-400" />
                            <p className="text-sm text-[var(--muted)]">Loading analytics…</p>
                        </div>
                    )}

                    {error && (
                        <div className="text-sm text-red-400 bg-red-900/20 border border-red-800/30 rounded-lg px-3 py-2">
                            {error}
                        </div>
                    )}

                    {!loading && summary && (
                        <>
                            {/* KPI cards */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Total Queries</p>
                                    <p className="text-xl font-semibold">{usage?.total_queries ?? 0}</p>
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Active Users</p>
                                    <p className="text-xl font-semibold">{usage?.unique_users ?? 0}</p>
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Citation Rate</p>
                                    <p className="text-xl font-semibold">{rag?.citation_coverage_rate ?? 0}%</p>
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Avg Sources / Query</p>
                                    <p className="text-xl font-semibold">{rag?.average_sources_per_query ?? 0}</p>
                                </div>
                            </div>

                            {/* Usage breakdown */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 space-y-3">
                                    <h3 className="text-sm font-semibold">Daily Queries</h3>
                                    <MiniBarList items={usage?.daily_queries ?? []} max={maxDailyQueries} />
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 space-y-3">
                                    <h3 className="text-sm font-semibold">Daily Active Users</h3>
                                    <MiniBarList
                                        items={usage?.daily_active_users ?? []}
                                        max={Math.max(1, (usage?.daily_active_users ?? []).reduce((m, d) => Math.max(m, d.count), 1))}
                                    />
                                </div>
                            </div>

                            {/* Feature usage + RAG quality */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 space-y-3">
                                    <h3 className="text-sm font-semibold">Feature Usage</h3>
                                    {(usage?.feature_usage ?? []).length === 0 ? (
                                        <p className="text-xs text-[var(--muted)]">No feature usage recorded.</p>
                                    ) : (
                                        <div className="space-y-2">
                                            {(usage?.feature_usage ?? []).map((row) => (
                                                <div key={row.event_type} className="flex items-center justify-between text-xs">
                                                    <span className="text-[var(--foreground)]">{row.feature}</span>
                                                    <span className="text-[var(--muted)]">{row.count}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 space-y-3">
                                    <h3 className="text-sm font-semibold">Daily Citation Rate</h3>
                                    <MiniRateList items={rag?.daily_quality ?? []} />
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">No-Answer Rate</p>
                                    <p className="text-lg font-semibold">{rag?.no_answer_rate ?? 0}%</p>
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Avg Answer Length</p>
                                    <p className="text-lg font-semibold">{rag?.average_answer_length_chars ?? 0}</p>
                                </div>
                                <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3">
                                    <p className="text-[11px] text-[var(--muted)]">Avg Relevance</p>
                                    <p className="text-lg font-semibold">
                                        {rag?.average_relevance_score ?? "N/A"}
                                    </p>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
