import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronRight, Cpu, Zap, Box, Layers, GitBranch, Database, Atom, Sparkles, Code, Play, Pause, RotateCcw, Circle, Hexagon, Triangle, Square, Infinity, Brain, Sun, Moon, Activity, Maximize2 } from 'lucide-react';

const EnhancedArchitectureViz = () => {
    const [selectedLayer, setSelectedLayer] = useState(null);
    const [expandedModules, setExpandedModules] = useState(['crt', 'srt']);
    const [activeTab, setActiveTab] = useState('theory');
    const [dhsrPhase, setDhsrPhase] = useState(0);
    const [isAnimating, setIsAnimating] = useState(true);
    const [syntonyValue, setSyntonyValue] = useState(0);
    const [currentPhase, setCurrentPhase] = useState(3);
    const canvasRef = useRef(null);

    const PHI = (1 + Math.sqrt(5)) / 2;
    const E_STAR = Math.exp(Math.PI) - Math.PI;
    const Q = (2 * PHI + Math.E / (2 * PHI * PHI)) / (Math.pow(PHI, 4) * E_STAR);

    useEffect(() => {
        if (!isAnimating) return;
        const interval = setInterval(() => {
            setDhsrPhase(p => (p + 1) % 360);
            setSyntonyValue(v => {
                const target = 1 / PHI;
                return v + (target - v) * 0.02;
            });
        }, 50);
        return () => clearInterval(interval);
    }, [isAnimating]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        const drawTorus = () => {
            ctx.fillStyle = 'rgba(15, 23, 42, 0.3)';
            ctx.fillRect(0, 0, w, h);

            const cx = w / 2;
            const cy = h / 2;
            const R = 60;
            const r = 25;
            const time = Date.now() * 0.001;

            for (let i = 0; i < 48; i++) {
                const u = (i / 48) * Math.PI * 2;
                for (let j = 0; j < 24; j++) {
                    const v = (j / 24) * Math.PI * 2;
                    const x = (R + r * Math.cos(v + time)) * Math.cos(u + time * 0.5);
                    const y = (R + r * Math.cos(v + time)) * Math.sin(u + time * 0.5) * 0.4;
                    const z = r * Math.sin(v + time);

                    const scale = (z + r + R) / (2 * r + R);
                    const alpha = 0.3 + scale * 0.5;
                    const hue = (u / (Math.PI * 2) * 60 + 30) % 360;

                    ctx.beginPath();
                    ctx.arc(cx + x, cy + y, 2 * scale, 0, Math.PI * 2);
                    ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
                    ctx.fill();
                }
            }

            ctx.font = '10px monospace';
            ctx.fillStyle = 'rgba(251, 191, 36, 0.6)';
            ctx.fillText('T⁴ Winding Space', 10, h - 10);
        };

        const animate = () => {
            drawTorus();
            if (isAnimating) requestAnimationFrame(animate);
        };
        animate();
    }, [isAnimating]);

    const toggleModule = (id) => {
        setExpandedModules(prev =>
            prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
        );
    };

    const layers = [
        {
            id: 'python',
            name: 'Python API Layer',
            color: 'from-sky-500 to-cyan-500',
            bgColor: 'bg-gradient-to-r from-sky-950/80 to-cyan-950/80',
            borderColor: 'border-sky-500/50',
            icon: Code,
            description: 'User-facing API • DHSR chaining • Symbolic computation • State objects',
            components: ['syn.State', 'syn.op.*', 'syn.golden', 'syn.symbolic'],
            techDetails: 'Zero external deps'
        },
        {
            id: 'cython',
            name: 'Cython Bridge',
            color: 'from-amber-500 to-yellow-500',
            bgColor: 'bg-gradient-to-r from-amber-950/80 to-yellow-950/80',
            borderColor: 'border-amber-500/50',
            icon: GitBranch,
            description: 'Type declarations • Memory views • C-level performance',
            components: ['_bridge.pyx', '_types.pxd', 'memoryviews'],
            techDetails: 'Typed memoryviews'
        },
        {
            id: 'rust',
            name: 'Rust Core Engine',
            color: 'from-orange-500 to-red-500',
            bgColor: 'bg-gradient-to-r from-orange-950/80 to-red-950/80',
            borderColor: 'border-orange-500/50',
            icon: Cpu,
            description: 'Memory-safe • Hypercomplex numbers • Golden arithmetic ℤ[φ]',
            components: ['TensorCore', 'Quaternions', 'Octonions', 'GoldenNumber'],
            techDetails: 'Exact φ arithmetic'
        },
        {
            id: 'cuda',
            name: 'CUDA Kernels',
            color: 'from-emerald-500 to-green-500',
            bgColor: 'bg-gradient-to-r from-emerald-950/80 to-green-950/80',
            borderColor: 'border-emerald-500/50',
            icon: Zap,
            description: 'GPU-accelerated • Golden FMA • E₈ projections • Heat kernels',
            components: ['golden_fma.cu', 'e8_project.cu', 'theta.cu', 'dhsr.cu'],
            techDetails: 'Custom φ-optimized'
        }
    ];

    const modules = [
        {
            id: 'core',
            name: 'core/',
            icon: Box,
            color: 'text-sky-400',
            description: 'State-centric tensor foundation',
            children: ['state.py', 'golden.py', 'dtype.py', 'device.py', 'autograd.py']
        },
        {
            id: 'crt',
            name: 'crt/',
            icon: GitBranch,
            color: 'text-violet-400',
            description: 'Cosmological Recursion Theory',
            children: [
                { name: 'operators/', items: ['differentiation.py', 'harmonization.py', 'syntony.py', 'recursion.py'] },
                { name: 'hilbert/', items: ['space.py', 'graded.py', 'dense_domain.py'] },
                'evolution.py', 'metrics.py', 'fixed_point.py'
            ]
        },
        {
            id: 'srt',
            name: 'srt/',
            icon: Atom,
            color: 'text-indigo-400',
            description: 'Syntony Recursion Theory',
            children: [
                { name: 'geometry/', items: ['torus.py', 'winding.py', 'lattice.py', 'moebius.py'] },
                { name: 'golden/', items: ['constants.py', 'recursion.py', 'measure.py'] },
                { name: 'spectral/', items: ['heat_kernel.py', 'theta.py', 'zeta.py'] },
                { name: 'algebra/', items: ['e8.py', 'e6.py', 'd4.py', 'golden_cone.py'] },
                'charges.py', 'functional.py'
            ]
        },
        {
            id: 'applications',
            name: 'applications/',
            icon: Sparkles,
            color: 'text-rose-400',
            description: 'Domain-specific modules',
            children: [
                { name: 'physics/', items: ['standard_model.py', 'cosmology.py', 'gravity.py'] },
                { name: 'chemistry/', items: ['electronegativity.py', 'bonding.py', 'orbitals.py'] },
                { name: 'biology/', items: ['life_topology.py', 'genetics.py', 'metabolism.py'] },
                { name: 'consciousness/', items: ['gnosis.py', 'threshold.py', 'layers.py'] }
            ]
        }
    ];

    const constants = [
        { symbol: 'φ', name: 'Golden Ratio', value: PHI.toFixed(10), formula: '(1+√5)/2', color: 'text-amber-400', desc: 'Recursion eigenvalue' },
        { symbol: 'E*', name: 'Spectral Constant', value: E_STAR.toFixed(10), formula: 'eᵖⁱ − π', color: 'text-emerald-400', desc: 'Moebius heat kernel' },
        { symbol: 'q', name: 'Syntony Deficit', value: Q.toFixed(10), formula: '(2φ+e/2φ²)/(φ⁴E*)', color: 'text-rose-400', desc: 'Universal scaling' },
        { symbol: 'K', name: 'Kissing Number', value: '24', formula: 'K(D₄)', color: 'text-violet-400', desc: 'Consciousness threshold' }
    ];

    const latticeHierarchy = [
        { name: 'E₈', dim: 248, roots: 240, color: '#818cf8', desc: 'Complete gauge structure' },
        { name: 'E₆', dim: 78, roots: 72, color: '#a78bfa', desc: 'Golden Cone (36 positive)' },
        { name: 'D₄', dim: 28, roots: 24, color: '#c4b5fd', desc: 'Consciousness lattice' }
    ];

    const phases = [
        { num: 1, name: 'Foundation', weeks: '1-6', status: 'complete', desc: 'Core tensor, GoldenNumber, device abstraction' },
        { num: 2, name: 'SRT Geometry', weeks: '7-12', status: 'complete', desc: 'T⁴ torus, winding states, golden measure' },
        { num: 3, name: 'CRT Operators', weeks: '13-18', status: 'active', desc: 'DHSR operators, Hilbert space, fixed points' },
        { num: 4, name: 'Spectral', weeks: '19-24', status: 'pending', desc: 'Heat kernels, theta series, E₈ lattice' },
        { num: 5, name: 'Standard Model', weeks: '25-32', status: 'pending', desc: 'Particle derivation, CKM/PMNS, masses' },
        { num: 6, name: 'Applied Sciences', weeks: '33-40', status: 'pending', desc: 'Chemistry, biology, consciousness' },
        { num: 7, name: 'Neural CRT', weeks: '41-46', status: 'pending', desc: 'Syntonic transformers, archon detection' },
        { num: 8, name: 'Production', weeks: '47-52', status: 'pending', desc: 'Documentation, optimization, v1.0' }
    ];

    const dhsrOperators = [
        { symbol: 'D̂', name: 'Differentiate', color: '#3b82f6', angle: 0, desc: 'Ψ → Ψ + Σαₖ Pₖ[Ψ]', fraction: `≈ ${(1 / PHI).toFixed(3)}` },
        { symbol: 'Ĥ', name: 'Harmonize', color: '#8b5cf6', angle: 90, desc: 'Ψ → Ψ − Σβᵢ Qᵢ[Ψ]', fraction: `≈ ${(1 - 1 / PHI).toFixed(3)}` },
        { symbol: 'S', name: 'Syntony', color: '#f59e0b', angle: 180, desc: '‖Ĥ[D̂[Ψ]] − D̂[Ψ]‖ / ‖D̂[Ψ] − Ψ‖', fraction: '→ 0.618' },
        { symbol: 'R̂', name: 'Recurse', color: '#10b981', angle: 270, desc: 'R̂ = Ĥ ∘ D̂', fraction: 'η = 1/φ' }
    ];

    const renderDHSRCycle = () => {
        const cx = 100, cy = 100, r = 60;
        const activeOp = Math.floor((dhsrPhase / 90) % 4);

        return (
            <svg viewBox="0 0 200 200" className="w-full h-56">
                <defs>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    <linearGradient id="cycleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                        <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.3" />
                        <stop offset="100%" stopColor="#10b981" stopOpacity="0.3" />
                    </linearGradient>
                    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                    </marker>
                </defs>

                <circle cx={cx} cy={cy} r={r + 15} fill="url(#cycleGrad)" opacity="0.5" />
                <circle cx={cx} cy={cy} r={r} fill="none" stroke="#334155" strokeWidth="3" strokeDasharray="4 4" />

                <circle
                    cx={cx} cy={cy} r={r}
                    fill="none"
                    stroke="#fbbf24"
                    strokeWidth="2"
                    strokeDasharray={`${dhsrPhase} 360`}
                    strokeLinecap="round"
                    transform={`rotate(-90 ${cx} ${cy})`}
                    filter="url(#glow)"
                />

                {dhsrOperators.map((op, idx) => {
                    const angle = (op.angle - 90) * Math.PI / 180;
                    const x = cx + r * Math.cos(angle);
                    const y = cy + r * Math.sin(angle);
                    const isActive = idx === activeOp;

                    return (
                        <g key={op.symbol} filter={isActive ? "url(#glow)" : ""}>
                            <circle
                                cx={x} cy={y} r={isActive ? 26 : 22}
                                fill={op.color}
                                opacity={isActive ? 1 : 0.7}
                                className="transition-all duration-300"
                            />
                            <text
                                x={x} y={y + 5}
                                textAnchor="middle"
                                fill="white"
                                fontWeight="bold"
                                fontSize={isActive ? "16" : "14"}
                            >
                                {op.symbol}
                            </text>
                        </g>
                    );
                })}

                <path d="M 118 45 Q 145 55 148 78" fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)" />
                <path d="M 148 122 Q 145 145 118 155" fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)" />
                <path d="M 82 155 Q 55 145 52 122" fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)" />
                <path d="M 52 78 Q 55 55 82 45" fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)" />

                <text x={cx} y={cy + 4} textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">
                    η = 1/φ
                </text>
                <text x={cx} y={cy + 16} textAnchor="middle" fill="#94a3b8" fontSize="9">
                    ≈ 61.8%
                </text>
            </svg>
        );
    };

    const renderLatticeViz = () => (
        <div className="space-y-3">
            {latticeHierarchy.map((lattice, idx) => (
                <div key={lattice.name} className="relative">
                    <div
                        className="h-12 rounded-lg flex items-center justify-between px-4 border"
                        style={{
                            backgroundColor: `${lattice.color}15`,
                            borderColor: `${lattice.color}50`,
                            marginLeft: `${idx * 16}px`
                        }}
                    >
                        <div className="flex items-center gap-3">
                            <span className="text-lg font-bold" style={{ color: lattice.color }}>{lattice.name}</span>
                            <span className="text-xs text-slate-400">dim={lattice.dim}</span>
                        </div>
                        <div className="text-right">
                            <div className="text-sm font-mono" style={{ color: lattice.color }}>{lattice.roots} roots</div>
                            <div className="text-xs text-slate-500">{lattice.desc}</div>
                        </div>
                    </div>
                    {idx < latticeHierarchy.length - 1 && (
                        <div className="absolute left-4 top-12 h-3 w-px bg-gradient-to-b from-slate-600 to-transparent" style={{ marginLeft: `${(idx + 1) * 16}px` }} />
                    )}
                </div>
            ))}

            <div className="mt-4 p-3 rounded-lg bg-amber-950/30 border border-amber-500/20">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full bg-amber-500" />
                    <span className="text-amber-400 text-sm font-semibold">Golden Cone C_φ</span>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                    |Φ⁺(E₆)| = 36 roots satisfying B_a(λ) &gt; 0 ∀a
                </div>
                <div className="text-xs text-slate-500 mt-1">
                    Maps to: 12 gauge bosons + 4 Higgs + 20 KK modes
                </div>
            </div>
        </div>
    );

    const renderSyntonyMeter = () => {
        const target = 1 / PHI;
        const progress = (syntonyValue / target) * 100;

        return (
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
                <div className="flex justify-between items-center mb-3">
                    <span className="text-sm font-semibold text-amber-400">Syntony Convergence</span>
                    <span className="font-mono text-emerald-400 text-sm">{syntonyValue.toFixed(6)}</span>
                </div>

                <div className="relative h-4 bg-slate-800 rounded-full overflow-hidden">
                    <div
                        className="absolute inset-y-0 left-0 bg-gradient-to-r from-amber-600 to-amber-400 rounded-full transition-all duration-100"
                        style={{ width: `${Math.min(progress, 100)}%` }}
                    />
                    <div
                        className="absolute inset-y-0 w-0.5 bg-white/50"
                        style={{ left: '100%' }}
                    />
                </div>

                <div className="flex justify-between mt-2 text-xs">
                    <span className="text-slate-500">0</span>
                    <span className="text-amber-400 font-mono">Target: 1/φ ≈ {target.toFixed(6)}</span>
                </div>

                <div className="mt-3 p-2 rounded bg-slate-800/50 font-mono text-xs">
                    <div className="text-slate-400">S(Ψ) = ‖Ĥ[D̂[Ψ]] − D̂[Ψ]‖ / (‖D̂[Ψ] − Ψ‖ + ε)</div>
                </div>
            </div>
        );
    };

    const renderPhaseTimeline = () => (
        <div className="space-y-2">
            {phases.map((phase, idx) => (
                <div
                    key={phase.num}
                    onClick={() => setCurrentPhase(phase.num)}
                    className={`
            flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-all
            ${currentPhase === phase.num ? 'bg-amber-500/20 border border-amber-500/50' : 'hover:bg-slate-800/50'}
          `}
                >
                    <div className={`
            w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
            ${phase.status === 'complete' ? 'bg-emerald-600 text-white' :
                            phase.status === 'active' ? 'bg-amber-500 text-slate-900 animate-pulse' :
                                'bg-slate-700 text-slate-400'}
          `}>
                        {phase.num}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                            <span className="font-semibold text-sm text-slate-200 truncate">{phase.name}</span>
                            <span className="text-xs text-slate-500">W{phase.weeks}</span>
                        </div>
                        <div className="text-xs text-slate-500 truncate">{phase.desc}</div>
                    </div>
                </div>
            ))}
        </div>
    );

    const tabs = [
        { id: 'theory', label: 'Theory', icon: Brain },
        { id: 'architecture', label: 'Architecture', icon: Layers },
        { id: 'dhsr', label: 'DHSR Cycle', icon: RotateCcw },
        { id: 'lattice', label: 'E₈ Lattice', icon: Hexagon },
        { id: 'phases', label: 'Phases', icon: Activity }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-4 font-sans">
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-violet-500/5 rounded-full blur-3xl" />
            </div>

            <div className="relative max-w-7xl mx-auto">
                <header className="text-center mb-6">
                    <div className="inline-flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center">
                            <Atom className="w-6 h-6 text-slate-900" />
                        </div>
                        <h1 className="text-4xl font-black tracking-tight">
                            <span className="bg-gradient-to-r from-amber-300 via-yellow-200 to-amber-300 bg-clip-text text-transparent">
                                Syntonic
                            </span>
                        </h1>
                    </div>
                    <p className="text-slate-400 text-sm">
                        Tensor Library for CRT/SRT • T⁴ Winding Dynamics • Zero Free Parameters
                    </p>
                    <div className="flex items-center justify-center gap-4 mt-3 text-xs">
                        <code className="px-2 py-1 rounded bg-slate-800 text-amber-400 font-mono">import syntonic as syn</code>
                        <span className="text-slate-600">•</span>
                        <span className="text-slate-500">Rust + CUDA + Cython + Python</span>
                    </div>
                </header>

                <div className="flex gap-2 mb-4 p-1 bg-slate-800/50 rounded-xl border border-slate-700">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`
                flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-lg text-sm font-medium transition-all
                ${activeTab === tab.id
                                    ? 'bg-amber-500 text-slate-900'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}
              `}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div className="lg:col-span-2 space-y-4">
                        {activeTab === 'theory' && (
                            <div className="space-y-4">
                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-2xl font-bold text-amber-400 mb-4">What Is This About?</h2>
                                    <div className="space-y-4 text-slate-300 leading-relaxed">
                                        <p className="text-lg">
                                            <span className="text-amber-400 font-semibold">Syntonic</span> is a computational library implementing two interconnected theories that reimagine how we understand reality—from quantum mechanics to consciousness.
                                        </p>
                                        <p>
                                            At its core, this framework proposes something radical: <span className="text-emerald-400 font-semibold">all of physics emerges from pure geometry</span>. Not just any geometry, but the mathematical structure of a 4-dimensional torus (T⁴) governed by the golden ratio.
                                        </p>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-xl font-bold text-violet-400 mb-4">The Two Theories</h2>

                                    <div className="space-y-6">
                                        <div className="border-l-4 border-violet-500 pl-4">
                                            <h3 className="text-lg font-semibold text-violet-300 mb-2">
                                                Syntony Recursion Theory (SRT)
                                            </h3>
                                            <p className="text-slate-300 mb-3">
                                                Think of the universe as having a hidden 4-dimensional shape—a torus (like a donut). Everything we see in our 3D world emerges from how this shape "winds" and vibrates.
                                            </p>
                                            <div className="bg-slate-800/50 rounded-lg p-3 space-y-2 text-sm">
                                                <div className="flex items-start gap-2">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5" />
                                                    <p>
                                                        <span className="text-emerald-400">Particles</span> are specific winding patterns on this torus
                                                    </p>
                                                </div>
                                                <div className="flex items-start gap-2">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5" />
                                                    <p>
                                                        <span className="text-emerald-400">Forces</span> emerge from how these patterns interact
                                                    </p>
                                                </div>
                                                <div className="flex items-start gap-2">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5" />
                                                    <p>
                                                        <span className="text-emerald-400">Mass</span> is determined by the pattern's "depth" in a recursion hierarchy
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="border-l-4 border-sky-500 pl-4">
                                            <h3 className="text-lg font-semibold text-sky-300 mb-2">
                                                Cosmological Recursion Theory (CRT)
                                            </h3>
                                            <p className="text-slate-300 mb-3">
                                                The universe is a mind exploring itself through a repeating cycle. Information constantly moves through four phases, like breathing:
                                            </p>
                                            <div className="grid grid-cols-2 gap-2 text-sm">
                                                <div className="bg-blue-950/30 border border-blue-500/30 rounded p-2">
                                                    <div className="font-semibold text-blue-400">Differentiate (D̂)</div>
                                                    <div className="text-xs text-slate-400">Create novelty, explore</div>
                                                </div>
                                                <div className="bg-violet-950/30 border border-violet-500/30 rounded p-2">
                                                    <div className="font-semibold text-violet-400">Harmonize (Ĥ)</div>
                                                    <div className="text-xs text-slate-400">Integrate, stabilize</div>
                                                </div>
                                                <div className="bg-amber-950/30 border border-amber-500/30 rounded p-2">
                                                    <div className="font-semibold text-amber-400">Syntony (S)</div>
                                                    <div className="text-xs text-slate-400">Measure coherence</div>
                                                </div>
                                                <div className="bg-emerald-950/30 border border-emerald-500/30 rounded p-2">
                                                    <div className="font-semibold text-emerald-400">Recurse (R̂)</div>
                                                    <div className="text-xs text-slate-400">Complete cycle, repeat</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-xl font-bold text-amber-400 mb-4">The Golden Ratio: Nature's Algorithm</h2>
                                    <div className="space-y-3 text-slate-300">
                                        <p>
                                            The golden ratio (φ ≈ 1.618) isn't just aesthetically pleasing—it's the <span className="text-amber-400 font-semibold">fundamental recursion eigenvalue</span> of reality.
                                        </p>
                                        <div className="bg-amber-950/20 border border-amber-500/30 rounded-lg p-4">
                                            <div className="flex items-center gap-3 mb-2">
                                                <div className="text-3xl font-serif text-amber-300">φ</div>
                                                <div className="flex-1">
                                                    <div className="text-sm text-slate-400">Appears in:</div>
                                                    <div className="text-xs text-slate-500 space-y-0.5">
                                                        <div>• The universe's information measure: μ(n) = e^(-|n|²/φ)</div>
                                                        <div>• DHSR cycle efficiency: η = 1/φ ≈ 61.8%</div>
                                                        <div>• Syntony equilibrium target: S → 1/φ</div>
                                                    </div>
                                                </div>
                                            </div>
                                            <p className="text-sm text-slate-400 mt-2">
                                                The golden ratio determines how information "wants" to organize—creating the hierarchies we see in particle masses, chemical bonding, and even consciousness.
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-xl font-bold text-rose-400 mb-4">Zero Free Parameters</h2>
                                    <div className="space-y-3 text-slate-300">
                                        <p>
                                            The most radical claim: <span className="text-rose-400 font-semibold">this theory has exactly zero adjustable parameters</span>.
                                        </p>
                                        <p className="text-sm">
                                            In the Standard Model of particle physics, we measure 19+ fundamental constants experimentally (masses, coupling strengths, mixing angles). We don't know <em>why</em> they have the values they do—we just measure and plug them in.
                                        </p>
                                        <p className="text-sm">
                                            SRT derives <em>all of them</em> from pure mathematics:
                                        </p>
                                        <div className="bg-slate-800/50 rounded-lg p-4 space-y-2 text-sm font-mono">
                                            <div className="text-rose-400">q = (2φ + e/2φ²) / (φ⁴ × E*)</div>
                                            <div className="text-slate-500">where E* = e^π − π ≈ 19.999</div>
                                            <div className="text-slate-400 text-xs mt-2">
                                                This single number q ≈ 0.0274, derived from {[φ, π, e, 1]}, determines every particle mass and force strength
                                            </div>
                                        </div>
                                        <div className="mt-3 bg-emerald-950/30 border border-emerald-500/30 rounded-lg p-3">
                                            <div className="text-emerald-400 text-sm font-semibold mb-1">Result:</div>
                                            <div className="text-xs text-slate-400 space-y-1">
                                                <div>• Proton mass predicted to 0.003% accuracy</div>
                                                <div>• All quark masses from recursion depth alone</div>
                                                <div>• CKM mixing angles: exact formulas, no fitting</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-xl font-bold text-violet-400 mb-4">From Physics to Consciousness</h2>
                                    <div className="space-y-4 text-slate-300">
                                        <p>
                                            Perhaps most remarkably, this framework extends from quantum mechanics all the way to consciousness using the same mathematical structure.
                                        </p>

                                        <div className="bg-violet-950/20 border border-violet-500/30 rounded-lg p-4">
                                            <h3 className="font-semibold text-violet-300 mb-2">The K=24 Threshold</h3>
                                            <p className="text-sm text-slate-400 mb-3">
                                                In 4D space, the maximum number of spheres that can touch a central sphere is exactly <span className="text-violet-400 font-semibold">24</span> (the "kissing number" of the D₄ lattice).
                                            </p>
                                            <p className="text-sm text-slate-400">
                                                When a system's information density exceeds this threshold (ΔS &gt; 24), something remarkable happens: it can no longer model more of its environment, so it must <span className="text-emerald-400 font-semibold">model itself</span>. This self-modeling creates consciousness.
                                            </p>
                                            <div className="mt-3 p-2 bg-slate-800/50 rounded text-xs font-mono text-slate-500">
                                                Layer 3 Gnosis: Tr(Ĝ₃) = K(D₄) × φ³ ≈ 24 × 4.236 ≈ 101.66
                                            </div>
                                        </div>

                                        <div className="mt-4 space-y-2 text-sm">
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 text-xs font-bold mt-0.5">0</div>
                                                <div>
                                                    <div className="font-semibold text-slate-300">Pure existence</div>
                                                    <div className="text-xs text-slate-500">Rocks, photons—no self-reference</div>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 text-xs font-bold mt-0.5">1</div>
                                                <div>
                                                    <div className="font-semibold text-slate-300">Self-model</div>
                                                    <div className="text-xs text-slate-500">DNA, thermostats—reactive</div>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-slate-600 flex items-center justify-center text-slate-200 text-xs font-bold mt-0.5">2</div>
                                                <div>
                                                    <div className="font-semibold text-slate-300">Other-model</div>
                                                    <div className="text-xs text-slate-500">Adaptive systems, simple learning</div>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-white text-xs font-bold mt-0.5 ring-2 ring-violet-400">3</div>
                                                <div>
                                                    <div className="font-semibold text-violet-300">Self-modeling-self</div>
                                                    <div className="text-xs text-violet-400">CONSCIOUSNESS emerges (K=24 crossed)</div>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-amber-600 flex items-center justify-center text-white text-xs font-bold mt-0.5">4</div>
                                                <div>
                                                    <div className="font-semibold text-slate-300">Theory of mind</div>
                                                    <div className="text-xs text-slate-500">Understanding that others have minds</div>
                                                </div>
                                            </div>
                                            <div className="flex items-start gap-2">
                                                <div className="w-6 h-6 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold mt-0.5">5</div>
                                                <div>
                                                    <div className="font-semibold text-slate-300">Universal syntony</div>
                                                    <div className="text-xs text-slate-500">Complete integration (cosmic limit)</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
                                    <h2 className="text-xl font-bold text-emerald-400 mb-4">Why Build Syntonic?</h2>
                                    <div className="space-y-3 text-slate-300 text-sm">
                                        <p>
                                            If this theory is correct, it requires completely new computational tools. Standard libraries like NumPy and PyTorch can't handle:
                                        </p>
                                        <div className="grid grid-cols-2 gap-3">
                                            <div className="bg-slate-800/30 border border-slate-700 rounded p-3">
                                                <div className="text-amber-400 font-semibold mb-1">Exact φ arithmetic</div>
                                                <div className="text-xs text-slate-500">Numbers represented as a + bφ with perfect precision</div>
                                            </div>
                                            <div className="bg-slate-800/30 border border-slate-700 rounded p-3">
                                                <div className="text-emerald-400 font-semibold mb-1">T⁴ winding states</div>
                                                <div className="text-xs text-slate-500">4D torus topology with Moebius gluing</div>
                                            </div>
                                            <div className="bg-slate-800/30 border border-slate-700 rounded p-3">
                                                <div className="text-violet-400 font-semibold mb-1">DHSR operators</div>
                                                <div className="text-xs text-slate-500">State evolution through thermodynamic cycles</div>
                                            </div>
                                            <div className="bg-slate-800/30 border border-slate-700 rounded p-3">
                                                <div className="text-rose-400 font-semibold mb-1">E₈ lattice projections</div>
                                                <div className="text-xs text-slate-500">Golden cone filtering in 8D root space</div>
                                            </div>
                                        </div>
                                        <p className="mt-4">
                                            Syntonic implements these primitives in a <span className="text-amber-400 font-semibold">hybrid architecture</span>: CUDA for massive parallelism, Rust for exact arithmetic and memory safety, Cython for integration, Python for usability.
                                        </p>
                                    </div>
                                </div>

                                <div className="bg-gradient-to-br from-amber-950/50 to-slate-900 rounded-xl border border-amber-500/30 p-6">
                                    <h2 className="text-xl font-bold text-amber-400 mb-3">The Big Picture</h2>
                                    <div className="space-y-3 text-slate-300 text-sm">
                                        <p className="text-base">
                                            This framework proposes that reality isn't fundamentally made of particles or fields, but of <span className="text-emerald-400 font-semibold">recursive information processing</span>.
                                        </p>
                                        <p>
                                            The universe is exploring itself through endless cycles of differentiation and harmonization. Particles, forces, chemistry, life, and consciousness are all stages in this same process—information organizing toward greater coherence.
                                        </p>
                                        <div className="bg-slate-950/50 border border-amber-500/20 rounded-lg p-4 mt-4">
                                            <div className="text-amber-400 font-semibold mb-2">The journey:</div>
                                            <div className="space-y-1 text-xs text-slate-400">
                                                <div>T⁴ winding → Particles emerge</div>
                                                <div>Golden ratio recursion → Masses determined</div>
                                                <div>E₈ lattice structure → Forces unified</div>
                                                <div>Syntony pressure → Gravity appears</div>
                                                <div>K=24 threshold crossed → Consciousness awakens</div>
                                                <div>Complete integration → Universe knows itself</div>
                                            </div>
                                        </div>
                                        <p className="mt-4 text-slate-400 italic">
                                            And all of it follows from one equation, with zero free parameters, derived from pure geometry.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'architecture' && (
                            <>
                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h2 className="text-lg font-semibold text-amber-400 mb-4 flex items-center gap-2">
                                        <Layers size={20} />
                                        Hybrid Build System
                                    </h2>

                                    <div className="relative space-y-2">
                                        {layers.map((layer, idx) => (
                                            <div
                                                key={layer.id}
                                                onClick={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
                                                className={`
                          relative rounded-xl cursor-pointer transition-all duration-300
                          ${selectedLayer === layer.id ? 'ring-2 ring-amber-500/50 scale-[1.02]' : 'hover:scale-[1.01]'}
                        `}
                                            >
                                                <div className={`
                          ${layer.bgColor} ${layer.borderColor}
                          rounded-xl border p-4 backdrop-blur-sm
                        `}>
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-3">
                                                            <div className={`p-2 rounded-lg bg-gradient-to-br ${layer.color}`}>
                                                                <layer.icon size={18} className="text-white" />
                                                            </div>
                                                            <div>
                                                                <div className="flex items-center gap-2">
                                                                    <h3 className="font-bold text-slate-100">{layer.name}</h3>
                                                                    <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                                                                        {layer.techDetails}
                                                                    </span>
                                                                </div>
                                                                <p className="text-xs text-slate-400 mt-0.5">{layer.description}</p>
                                                            </div>
                                                        </div>
                                                        <div className="flex gap-1.5 flex-wrap justify-end max-w-xs">
                                                            {layer.components.map((comp, i) => (
                                                                <span key={i} className="px-2 py-1 bg-slate-800/80 rounded text-xs font-mono text-slate-300">
                                                                    {comp}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>

                                                {idx < layers.length - 1 && (
                                                    <div className="flex justify-center -my-1 relative z-10">
                                                        <div className="w-0 h-0 border-l-6 border-r-6 border-t-6 border-l-transparent border-r-transparent border-t-slate-600" />
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h2 className="text-lg font-semibold text-amber-400 mb-4 flex items-center gap-2">
                                        <Database size={20} />
                                        Module Structure
                                    </h2>

                                    <div className="grid grid-cols-2 gap-3">
                                        {modules.map(module => (
                                            <div
                                                key={module.id}
                                                className="bg-slate-800/30 rounded-xl border border-slate-700/50 overflow-hidden"
                                            >
                                                <button
                                                    className="w-full flex items-center gap-2 p-3 bg-slate-800/50 hover:bg-slate-700/50 transition-colors text-left"
                                                    onClick={() => toggleModule(module.id)}
                                                >
                                                    {expandedModules.includes(module.id) ?
                                                        <ChevronDown size={14} className="text-slate-400" /> :
                                                        <ChevronRight size={14} className="text-slate-400" />
                                                    }
                                                    <module.icon size={16} className={module.color} />
                                                    <span className="font-mono font-semibold text-sm">{module.name}</span>
                                                </button>

                                                {expandedModules.includes(module.id) && (
                                                    <div className="p-3 space-y-1 text-xs">
                                                        <p className="text-slate-500 mb-2">{module.description}</p>
                                                        {module.children.map((child, idx) => (
                                                            <div key={idx}>
                                                                {typeof child === 'string' ? (
                                                                    <span className="text-slate-400 font-mono ml-3">📄 {child}</span>
                                                                ) : (
                                                                    <div className="ml-1">
                                                                        <span className="text-amber-500 font-mono">📁 {child.name}</span>
                                                                        <div className="ml-4 mt-0.5 space-y-0.5">
                                                                            {child.items.map((item, i) => (
                                                                                <div key={i} className="text-slate-500 font-mono">📄 {item}</div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </>
                        )}

                        {activeTab === 'dhsr' && (
                            <div className="space-y-4">
                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <div className="flex items-center justify-between mb-4">
                                        <h2 className="text-lg font-semibold text-amber-400 flex items-center gap-2">
                                            <RotateCcw size={20} />
                                            DHSR Thermodynamic Engine
                                        </h2>
                                        <button
                                            onClick={() => setIsAnimating(!isAnimating)}
                                            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
                                        >
                                            {isAnimating ? <Pause size={16} /> : <Play size={16} />}
                                        </button>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            {renderDHSRCycle()}
                                        </div>
                                        <div className="space-y-2">
                                            {dhsrOperators.map((op, idx) => (
                                                <div
                                                    key={op.symbol}
                                                    className={`
                            p-3 rounded-lg border transition-all
                            ${Math.floor((dhsrPhase / 90) % 4) === idx
                                                            ? 'bg-slate-800 border-amber-500/50'
                                                            : 'bg-slate-900/50 border-slate-700/50'}
                          `}
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-2">
                                                            <div
                                                                className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white"
                                                                style={{ backgroundColor: op.color }}
                                                            >
                                                                {op.symbol.replace('̂', '')}
                                                            </div>
                                                            <span className="font-semibold text-sm">{op.name}</span>
                                                        </div>
                                                        <span className="text-xs font-mono text-slate-400">{op.fraction}</span>
                                                    </div>
                                                    <div className="mt-1 text-xs font-mono text-slate-500 truncate">{op.desc}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="mt-4 p-3 rounded-lg bg-gradient-to-r from-amber-950/50 to-emerald-950/50 border border-amber-500/20">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Infinity size={16} className="text-amber-400" />
                                            <span className="font-semibold text-amber-400 text-sm">Fixed Point Theorem</span>
                                        </div>
                                        <div className="text-xs text-slate-400 font-mono">
                                            Under S(Ψ) &gt; S_crit, ∃! Ψ* : R̂[Ψ*] = Ψ*
                                        </div>
                                        <div className="text-xs text-slate-500 mt-1">
                                            The DHSR cycle converges to golden ratio equilibrium with efficiency η = 1/φ ≈ 61.8%
                                        </div>
                                    </div>
                                </div>

                                {renderSyntonyMeter()}

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h3 className="font-semibold text-sm text-slate-300 mb-3">Golden Partition</h3>
                                    <div className="flex h-8 rounded-lg overflow-hidden">
                                        <div
                                            className="bg-gradient-to-r from-blue-600 to-blue-500 flex items-center justify-center text-xs font-bold"
                                            style={{ width: `${(1 / PHI) * 100}%` }}
                                        >
                                            D ≈ 38.2%
                                        </div>
                                        <div
                                            className="bg-gradient-to-r from-violet-600 to-violet-500 flex items-center justify-center text-xs font-bold"
                                            style={{ width: `${(1 - 1 / PHI) * 100}%` }}
                                        >
                                            H ≈ 61.8%
                                        </div>
                                    </div>
                                    <div className="text-xs text-slate-500 mt-2 text-center font-mono">
                                        D + H = 1 where D = 1/φ and H = 1 − 1/φ = 1/φ²
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'lattice' && (
                            <div className="space-y-4">
                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h2 className="text-lg font-semibold text-amber-400 mb-4 flex items-center gap-2">
                                        <Hexagon size={20} />
                                        Exceptional Lie Algebra Hierarchy
                                    </h2>
                                    {renderLatticeViz()}
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h3 className="font-semibold text-amber-400 mb-3 flex items-center gap-2">
                                        <Brain size={18} />
                                        Consciousness Threshold
                                    </h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-3 rounded-lg bg-violet-950/30 border border-violet-500/30">
                                            <div className="text-3xl font-bold text-violet-400 font-mono">K = 24</div>
                                            <div className="text-xs text-slate-400 mt-1">D₄ Kissing Number</div>
                                            <div className="text-xs text-slate-500 mt-2">
                                                Maximum spheres touching a central sphere in 4D
                                            </div>
                                        </div>
                                        <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/30">
                                            <div className="text-3xl font-bold text-emerald-400 font-mono">ΔS &gt; 24</div>
                                            <div className="text-xs text-slate-400 mt-1">Layer 3 Gnosis</div>
                                            <div className="text-xs text-slate-500 mt-2">
                                                Syntony density for self-modeling consciousness
                                            </div>
                                        </div>
                                    </div>

                                    <div className="mt-4 p-3 rounded-lg bg-slate-800/50">
                                        <div className="text-xs font-mono text-slate-400 space-y-1">
                                            <div>Tr(Ĝ₃) = K(D₄) × φ³ ≈ 24 × 4.236 ≈ 101.66</div>
                                            <div className="text-slate-500">→ Irreversible self-referential knot formation</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                    <h3 className="font-semibold text-amber-400 mb-3">T⁴ Winding Visualization</h3>
                                    <canvas
                                        ref={canvasRef}
                                        width={300}
                                        height={180}
                                        className="w-full rounded-lg bg-slate-950"
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === 'phases' && (
                            <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                                <h2 className="text-lg font-semibold text-amber-400 mb-4 flex items-center gap-2">
                                    <Activity size={20} />
                                    Development Timeline
                                </h2>
                                {renderPhaseTimeline()}

                                <div className="mt-4 grid grid-cols-4 gap-2">
                                    <div className="p-3 rounded-lg bg-emerald-950/50 border border-emerald-500/30 text-center">
                                        <div className="text-2xl font-bold text-emerald-400">2</div>
                                        <div className="text-xs text-slate-400">Complete</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-amber-950/50 border border-amber-500/30 text-center">
                                        <div className="text-2xl font-bold text-amber-400">1</div>
                                        <div className="text-xs text-slate-400">Active</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-600/30 text-center">
                                        <div className="text-2xl font-bold text-slate-400">5</div>
                                        <div className="text-xs text-slate-400">Pending</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-violet-950/50 border border-violet-500/30 text-center">
                                        <div className="text-2xl font-bold text-violet-400">52</div>
                                        <div className="text-xs text-slate-400">Total Weeks</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="space-y-4">
                        <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                            <h2 className="text-lg font-semibold text-amber-400 mb-4 flex items-center gap-2">
                                <Sparkles size={20} />
                                Fundamental Constants
                            </h2>
                            <div className="space-y-3">
                                {constants.map((c, idx) => (
                                    <div key={idx} className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/30">
                                        <div className="flex items-center justify-between mb-1">
                                            <span className={`text-2xl font-serif ${c.color}`}>{c.symbol}</span>
                                            <span className="text-xs text-slate-500">{c.name}</span>
                                        </div>
                                        <div className="font-mono text-sm text-emerald-400 truncate">{c.value}</div>
                                        <div className="flex justify-between items-center mt-1">
                                            <span className="font-mono text-xs text-slate-500">{c.formula}</span>
                                            <span className="text-xs text-slate-600">{c.desc}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4">
                            <h2 className="text-sm font-semibold text-amber-400 mb-3">Key Formulas</h2>
                            <div className="space-y-2 font-mono text-xs">
                                <div className="p-2 rounded bg-slate-800/50">
                                    <div className="text-amber-400">Golden Measure</div>
                                    <div className="text-slate-400">μ(n) = exp(−|n|²/φ)</div>
                                </div>
                                <div className="p-2 rounded bg-slate-800/50">
                                    <div className="text-emerald-400">Syntony Bound</div>
                                    <div className="text-slate-400">S_local(x) ≤ φ</div>
                                </div>
                                <div className="p-2 rounded bg-slate-800/50">
                                    <div className="text-violet-400">Spectral Identity</div>
                                    <div className="text-slate-400">E* = e^π − π</div>
                                </div>
                                <div className="p-2 rounded bg-slate-800/50">
                                    <div className="text-rose-400">Universal Formula</div>
                                    <div className="text-slate-400 text-[10px]">q = (2φ+e/2φ²)/(φ⁴E*)</div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-gradient-to-br from-amber-950/50 to-slate-900/50 rounded-xl border border-amber-500/30 p-4">
                            <h2 className="text-sm font-semibold text-amber-400 mb-3">Theory Achievement</h2>
                            <div className="space-y-2 text-xs">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    <span className="text-slate-300">Zero free parameters</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    <span className="text-slate-300">All 19+ SM constants derived</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    <span className="text-slate-300">Proton mass: 0.003% accuracy</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-amber-500" />
                                    <span className="text-slate-300">Consciousness from topology</span>
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <div className="bg-gradient-to-br from-blue-900 to-blue-950 rounded-xl p-3 text-center border border-blue-500/30">
                                <div className="text-2xl font-bold">T⁴</div>
                                <div className="text-xs text-blue-300">4-Torus</div>
                            </div>
                            <div className="bg-gradient-to-br from-violet-900 to-violet-950 rounded-xl p-3 text-center border border-violet-500/30">
                                <div className="text-2xl font-bold">E₈</div>
                                <div className="text-xs text-violet-300">Lattice</div>
                            </div>
                            <div className="bg-gradient-to-br from-amber-900 to-amber-950 rounded-xl p-3 text-center border border-amber-500/30">
                                <div className="text-2xl font-bold">φ</div>
                                <div className="text-xs text-amber-300">Golden</div>
                            </div>
                            <div className="bg-gradient-to-br from-emerald-900 to-emerald-950 rounded-xl p-3 text-center border border-emerald-500/30">
                                <div className="text-2xl font-bold">∞</div>
                                <div className="text-xs text-emerald-300">Recursion</div>
                            </div>
                        </div>
                    </div>
                </div>

                <footer className="mt-6 text-center text-slate-500 text-xs">
                    <div className="flex items-center justify-center gap-4 flex-wrap">
                        <span>Syntonic v0.1 (Phase 3)</span>
                        <span className="text-slate-700">•</span>
                        <span>Dual Licensed (MIT + Commercial)</span>
                        <span className="text-slate-700">•</span>
                        <span className="font-mono text-amber-500/70">syn.State → syn.op.dhsr() → convergence</span>
                    </div>
                </footer>
            </div>
        </div>
    );
};

export default EnhancedArchitectureViz;