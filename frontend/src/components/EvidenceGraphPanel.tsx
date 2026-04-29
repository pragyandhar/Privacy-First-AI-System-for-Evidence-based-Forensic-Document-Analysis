"use client";

import { useState } from "react";
import {
    fetchEvidenceLinks,
    type GraphNode,
    type GraphEdge,
} from "@/lib/api";
import { Network, Loader2, X, CheckCircle } from "lucide-react";
import { useEffect, useRef } from "react";

interface Props {
    open: boolean;
    onClose: () => void;
}

/* ---- Tiny force-directed layout engine -------------------------------- */

interface SimNode {
    id: string;
    type: string;
    source: string;
    x: number;
    y: number;
    vx: number;
    vy: number;
}

function layoutGraph(
    nodes: GraphNode[],
    edges: GraphEdge[],
    width: number,
    height: number,
    iterations = 120,
): SimNode[] {
    if (nodes.length === 0) return [];

    // Initialize nodes in a circle
    const simNodes: SimNode[] = nodes.map((n, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        const r = Math.min(width, height) * 0.3;
        return {
            ...n,
            x: width / 2 + r * Math.cos(angle),
            y: height / 2 + r * Math.sin(angle),
            vx: 0,
            vy: 0,
        };
    });

    const idMap = new Map(simNodes.map((n) => [n.id, n]));

    for (let iter = 0; iter < iterations; iter++) {
        const alpha = 1 - iter / iterations;

        // Repulsion between all nodes
        for (let i = 0; i < simNodes.length; i++) {
            for (let j = i + 1; j < simNodes.length; j++) {
                const a = simNodes[i],
                    b = simNodes[j];
                let dx = b.x - a.x || 0.1;
                let dy = b.y - a.y || 0.1;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = (800 * alpha) / (dist * dist);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                a.vx -= fx;
                a.vy -= fy;
                b.vx += fx;
                b.vy += fy;
            }
        }

        // Attraction along edges
        for (const edge of edges) {
            const a = idMap.get(edge.from);
            const b = idMap.get(edge.to);
            if (!a || !b) continue;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = dist * 0.005 * alpha;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            a.vx += fx;
            a.vy += fy;
            b.vx -= fx;
            b.vy -= fy;
        }

        // Center gravity
        for (const n of simNodes) {
            n.vx += (width / 2 - n.x) * 0.001 * alpha;
            n.vy += (height / 2 - n.y) * 0.001 * alpha;
        }

        // Apply velocities with damping
        for (const n of simNodes) {
            n.vx *= 0.6;
            n.vy *= 0.6;
            n.x += n.vx;
            n.y += n.vy;
            // Clamp to bounds
            n.x = Math.max(60, Math.min(width - 60, n.x));
            n.y = Math.max(40, Math.min(height - 40, n.y));
        }
    }

    return simNodes;
}

/* ---- Colour by node type ----------------------------------------------- */

const typeColors: Record<string, string> = {
    person: "#818cf8",
    event: "#f59e0b",
    evidence: "#22c55e",
    document: "#ec4899",
    location: "#06b6d4",
    entity: "#a78bfa",
};

const strengthColors: Record<string, string> = {
    strong: "#22c55e",
    moderate: "#f59e0b",
    weak: "#ef4444",
};

/* ---- Component --------------------------------------------------------- */

export default function EvidenceGraphPanel({ open, onClose }: Props) {
    const [nodes, setNodes] = useState<GraphNode[]>([]);
    const [edges, setEdges] = useState<GraphEdge[]>([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [hasLoaded, setHasLoaded] = useState(false);
    const svgRef = useRef<SVGSVGElement>(null);
    const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);

    const W = 700;
    const H = 480;

    const handleAnalyze = async () => {
        setLoading(true);
        setMessage(null);
        try {
            const res = await fetchEvidenceLinks();
            setNodes(res.nodes);
            setEdges(res.edges);
            setMessage(res.message);
            setHasLoaded(true);
        } catch (err: any) {
            setMessage(err.message ?? "Evidence linking failed");
        } finally {
            setLoading(false);
        }
    };

    if (!open) return null;

    const simNodes = hasLoaded ? layoutGraph(nodes, edges, W, H) : [];
    const idMap = new Map(simNodes.map((n) => [n.id, n]));

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-4xl max-h-[90vh] bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <Network size={18} className="text-green-400" />
                        <h2 className="text-sm font-semibold">Forensic Evidence Linking</h2>
                    </div>
                    <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
                    {!hasLoaded && !loading && (
                        <div className="text-center space-y-3 py-8">
                            <Network size={40} className="mx-auto text-green-400/50" />
                            <p className="text-sm text-[var(--muted)]">
                                Analyze connections and relationships across documents to build a forensic evidence graph.
                            </p>
                            <button
                                onClick={handleAnalyze}
                                className="px-5 py-2.5 rounded-xl bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
                            >
                                Show Forensic Linking
                            </button>
                        </div>
                    )}

                    {loading && (
                        <div className="flex flex-col items-center gap-3 py-12">
                            <Loader2 size={32} className="animate-spin text-green-400" />
                            <p className="text-sm text-[var(--muted)]">Analyzing cross-document evidence links...</p>
                        </div>
                    )}

                    {hasLoaded && !loading && (
                        <>
                            {message && (
                                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-900/20 text-green-300 text-xs">
                                    <CheckCircle size={14} />
                                    <span>{message}</span>
                                </div>
                            )}

                            {/* Legend */}
                            <div className="flex flex-wrap gap-3 text-[11px]">
                                {Object.entries(typeColors).map(([type, color]) => (
                                    <span key={type} className="flex items-center gap-1.5">
                                        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                                        {type}
                                    </span>
                                ))}
                                <span className="mx-2 text-[var(--border)]">|</span>
                                {Object.entries(strengthColors).map(([str, color]) => (
                                    <span key={str} className="flex items-center gap-1.5">
                                        <span className="w-6 h-0.5" style={{ backgroundColor: color }} />
                                        {str}
                                    </span>
                                ))}
                            </div>

                            {/* SVG Graph */}
                            {nodes.length > 0 ? (
                                <div className="border border-[var(--border)] rounded-xl overflow-hidden bg-[var(--background)]">
                                    <svg
                                        ref={svgRef}
                                        viewBox={`0 0 ${W} ${H}`}
                                        className="w-full"
                                        style={{ maxHeight: "60vh" }}
                                    >
                                        {/* Edges */}
                                        {edges.map((edge, i) => {
                                            const a = idMap.get(edge.from);
                                            const b = idMap.get(edge.to);
                                            if (!a || !b) return null;
                                            const color = strengthColors[edge.strength] ?? "#666";
                                            const isHovered = hoveredEdge === i;
                                            return (
                                                <g key={`edge-${i}`}>
                                                    <line
                                                        x1={a.x}
                                                        y1={a.y}
                                                        x2={b.x}
                                                        y2={b.y}
                                                        stroke={color}
                                                        strokeWidth={isHovered ? 3 : 1.5}
                                                        strokeOpacity={isHovered ? 1 : 0.6}
                                                        onMouseEnter={() => setHoveredEdge(i)}
                                                        onMouseLeave={() => setHoveredEdge(null)}
                                                        style={{ cursor: "pointer" }}
                                                    />
                                                    {/* Edge label */}
                                                    {isHovered && (
                                                        <text
                                                            x={(a.x + b.x) / 2}
                                                            y={(a.y + b.y) / 2 - 8}
                                                            textAnchor="middle"
                                                            fill="#e5e7eb"
                                                            fontSize="10"
                                                            fontWeight="500"
                                                        >
                                                            {edge.relationship}
                                                        </text>
                                                    )}
                                                </g>
                                            );
                                        })}

                                        {/* Nodes */}
                                        {simNodes.map((node) => {
                                            const color = typeColors[node.type] ?? "#a78bfa";
                                            return (
                                                <g key={node.id}>
                                                    <circle
                                                        cx={node.x}
                                                        cy={node.y}
                                                        r={18}
                                                        fill={color}
                                                        fillOpacity={0.25}
                                                        stroke={color}
                                                        strokeWidth={2}
                                                    />
                                                    <circle cx={node.x} cy={node.y} r={6} fill={color} />
                                                    <text
                                                        x={node.x}
                                                        y={node.y + 28}
                                                        textAnchor="middle"
                                                        fill="#e5e7eb"
                                                        fontSize="10"
                                                        fontWeight="500"
                                                    >
                                                        {node.id.length > 18 ? node.id.slice(0, 16) + "…" : node.id}
                                                    </text>
                                                </g>
                                            );
                                        })}
                                    </svg>
                                </div>
                            ) : (
                                <div className="text-center py-8 text-sm text-[var(--muted)]">
                                    No evidence connections found.
                                </div>
                            )}

                            <button
                                onClick={handleAnalyze}
                                className="w-full px-4 py-2 rounded-xl bg-[var(--card-hover)] text-sm hover:bg-green-600/20 transition-colors text-[var(--muted)]"
                            >
                                Re-analyze
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
