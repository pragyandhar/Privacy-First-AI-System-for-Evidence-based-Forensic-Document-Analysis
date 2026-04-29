"use client";

import { useState } from "react";
import { fetchAnomalies, type AnomalyAlert } from "@/lib/api";
import { ShieldAlert, Loader2, X, AlertTriangle, Info, AlertOctagon, CheckCircle } from "lucide-react";

interface Props {
    open: boolean;
    onClose: () => void;
}

export default function AnomalyPanel({ open, onClose }: Props) {
    const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [severityCounts, setSeverityCounts] = useState<Record<string, number>>({});
    const [hasLoaded, setHasLoaded] = useState(false);

    const handleDetect = async () => {
        setLoading(true);
        setMessage(null);
        try {
            const res = await fetchAnomalies();
            setAlerts(res.alerts);
            setMessage(res.message);
            setSeverityCounts(res.severity_counts);
            setHasLoaded(true);
        } catch (err: any) {
            setMessage(err.message ?? "Anomaly detection failed");
        } finally {
            setLoading(false);
        }
    };

    if (!open) return null;

    const severityConfig: Record<string, { icon: React.ReactNode; color: string; bg: string }> = {
        critical: {
            icon: <AlertOctagon size={16} />,
            color: "text-red-400",
            bg: "bg-red-900/20 border-red-800/30",
        },
        warning: {
            icon: <AlertTriangle size={16} />,
            color: "text-yellow-400",
            bg: "bg-yellow-900/20 border-yellow-800/30",
        },
        info: {
            icon: <Info size={16} />,
            color: "text-blue-400",
            bg: "bg-blue-900/20 border-blue-800/30",
        },
    };

    const typeLabels: Record<string, string> = {
        inconsistency: "Inconsistency",
        missing_data: "Missing Data",
        unusual_pattern: "Unusual Pattern",
        suspicious: "Suspicious",
        red_flag: "Red Flag",
        information: "Information",
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-2xl max-h-[80vh] bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <ShieldAlert size={18} className="text-yellow-400" />
                        <h2 className="text-sm font-semibold">Smart Alerts & Anomaly Detection</h2>
                    </div>
                    <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
                    {!hasLoaded && !loading && (
                        <div className="text-center space-y-3 py-8">
                            <ShieldAlert size={40} className="mx-auto text-yellow-400/50" />
                            <p className="text-sm text-[var(--muted)]">
                                Analyze documents for anomalies, inconsistencies, and suspicious patterns.
                            </p>
                            <button
                                onClick={handleDetect}
                                className="px-5 py-2.5 rounded-xl bg-yellow-600 hover:bg-yellow-500 text-white text-sm font-medium transition-colors"
                            >
                                Detect Anomalies
                            </button>
                        </div>
                    )}

                    {loading && (
                        <div className="flex flex-col items-center gap-3 py-12">
                            <Loader2 size={32} className="animate-spin text-yellow-400" />
                            <p className="text-sm text-[var(--muted)]">Scanning documents for anomalies...</p>
                        </div>
                    )}

                    {hasLoaded && !loading && (
                        <>
                            {/* Summary badges */}
                            <div className="flex gap-2 flex-wrap">
                                {Object.entries(severityCounts).map(([sev, count]) => {
                                    const cfg = severityConfig[sev] ?? severityConfig.info;
                                    return (
                                        <span
                                            key={sev}
                                            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full ${cfg.color} ${cfg.bg} border`}
                                        >
                                            {cfg.icon}
                                            {count} {sev}
                                        </span>
                                    );
                                })}
                            </div>

                            {message && (
                                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-600/10 text-brand-300 text-xs">
                                    <CheckCircle size={14} />
                                    <span>{message}</span>
                                </div>
                            )}

                            {/* Alert cards */}
                            {alerts.map((alert, i) => {
                                const cfg = severityConfig[alert.severity] ?? severityConfig.info;
                                return (
                                    <div key={i} className={`border rounded-xl p-4 space-y-2 ${cfg.bg}`}>
                                        <div className="flex items-center gap-2">
                                            <span className={cfg.color}>{cfg.icon}</span>
                                            <span className={`text-sm font-semibold ${cfg.color}`}>{alert.title}</span>
                                            <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-[var(--background)] text-[var(--muted)]">
                                                {typeLabels[alert.type] ?? alert.type}
                                            </span>
                                        </div>
                                        <p className="text-sm text-[var(--foreground)]">{alert.description}</p>
                                        <div className="flex gap-4 text-[11px] text-[var(--muted)]">
                                            <span>Source: {alert.source}</span>
                                        </div>
                                        {alert.recommendation && (
                                            <div className="text-xs px-3 py-2 rounded-lg bg-[var(--background)] text-[var(--muted)]">
                                                <strong>Recommendation:</strong> {alert.recommendation}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}

                            <button
                                onClick={handleDetect}
                                className="w-full mt-2 px-4 py-2 rounded-xl bg-[var(--card-hover)] text-sm hover:bg-yellow-600/20 transition-colors text-[var(--muted)]"
                            >
                                Re-scan
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
