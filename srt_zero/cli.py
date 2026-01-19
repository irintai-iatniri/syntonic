"""
SRT-Zero: Command-Line Interface
================================

Usage:
    python -m srt_zero.cli derive proton
    python -m srt_zero.cli mine 125100 --tolerance 0.05
    python -m srt_zero.cli validate
    python -m srt_zero.cli list
"""

from __future__ import annotations

import argparse
import sys
import json
import os
from typing import Optional

from .engine import DerivationEngine, MassMiner
from .catalog import get_particle, list_particles, ParticleType, FormulaType, CATALOG
from .hierarchy import E_STAR, Q, PHI


def cmd_derive(args: argparse.Namespace) -> int:
    """Derive a particle's mass."""
    engine = DerivationEngine()
    
    try:
        result = engine.derive(args.particle)
        config = get_particle(args.particle)
        
        print(f"\n{'='*60}")
        print(f"SRT-Zero: {config.name} ({config.symbol})")
        print(f"{'='*60}")
        print(f"\nFormula Type: {config.formula_type.name}")
        if config.base_integer_N:
            print(f"Base Integer N: {config.base_integer_N}")
        if config.corrections:
            corrs = [f"(1 {'+' if s>0 else '-'} q/{d})" for d, s in config.corrections]
            print(f"Corrections: {''.join(corrs)}")
        if config.special_corrections:
            print(f"Special: {config.special_corrections}")
        
        print(f"\nTree-level:  {float(result.tree_value):.6f} {config.pdg_unit}")
        print(f"Final:       {float(result.final_value):.6f} {config.pdg_unit}")
        print(f"PDG Value:   {config.pdg_value:.6f} ± {config.pdg_uncertainty} {config.pdg_unit}")
        
        exp = config.pdg_value
        pred = float(result.final_value)
        
        # Handle unmeasured quantities (PDG=0.0)
        if abs(exp) < 1e-12 and config.pdg_uncertainty == 0.0:
            print(f"\nThis is an SRT PREDICTION (no experimental value available)")
        elif abs(exp) < 1e-12:
            print(f"\nDeviation: N/A (experimental value is 0)")
        else:
            error = 100 * abs(pred - exp) / exp
            print(f"\nDeviation:   {error:.4f}%")
        
        if result.steps:
            print(f"\nCorrection Steps:")
            for i, step in enumerate(result.steps, 1):
                if isinstance(step, dict):
                    print(f"  {i}. {step.get('description', step.get('type', 'unknown'))}")
        
        if config.notes:
            print(f"\nNotes: {config.notes}")
        
        return 0
        
    except KeyError as e:
        print(f"Error: Unknown particle '{args.particle}'")
        print(f"Use 'python -m srt_zero.cli list' to see available particles")
        return 1


def cmd_mine(args: argparse.Namespace) -> int:
    """Mine for formulas matching a target mass."""
    engine = DerivationEngine()
    miner = MassMiner(engine)
    
    print(f"\n{'='*60}")
    print(f"SRT-Zero Mass Miner: Target = {args.mass} MeV")
    print(f"Tolerance: {args.tolerance}%")
    print(f"{'='*60}")
    
    # Try E* template
    print(f"\n>> Searching E* × N × (1 ± q/divisor)...")
    matches = miner.mine_E_star(args.mass, tolerance_percent=args.tolerance)
    
    if matches:
        print(f"\nFound {len(matches)} matches (top 10):")
        print(f"\n{'N':<10} {'Correction':<15} {'Sign':<5} {'Mass':<12} {'Error':<10}")
        print("-" * 55)
        for m in matches[:10]:
            print(f"{m['integer']:<10.1f} {m['correction']:<15} {m['sign']:<5} {m['mass']:<12.3f} {m['error_percent']:<10.6f}%")
    else:
        print("No E* matches found within tolerance.")
    
    # Try proton ratio
    print(f"\n>> Searching m_p × (1 ± n·q)...")
    # Note: mine_from_proton method not implemented yet
    print("Proton-based mining not yet implemented.")
    
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation suite."""
    engine = DerivationEngine()
    
    print(f"\n{'='*60}")
    print("SRT-Zero Validation Suite")
    print(f"{'='*60}")
    print(f"\nE* = {float(E_STAR):.15f}")
    print(f"q  = {float(Q):.15f}")
    print(f"φ  = {float(PHI):.15f}")
    
    results_data = []
    particles = list(CATALOG.keys())
    
    # Filter to unique particles (remove aliases)
    if args.unique:
        seen = set()
        unique_particles = []
        for name in particles:
            config = get_particle(name)
            # Use (name, pdg_value) as unique key
            key = (config.name, config.pdg_value)
            if key not in seen:
                seen.add(key)
                unique_particles.append(name)
        particles = unique_particles
        print(f"\n(Showing {len(particles)} unique particles, excluding aliases)")
    
    print(f"\n{'Particle':<15} {'Predicted':>12} {'PDG':>12} {'Error':>10} {'Status':>8}")
    print("-" * 60)
    
    passed = 0
    failed = 0
    predictions = 0
    
    for name in particles:
        try:
            result = engine.derive(name)
            config = get_particle(name)
            
            pred = float(result.final_value)
            exp = config.pdg_value
            
            # Unit conversion
            if config.formula_type in [FormulaType.COSMOLOGY_SPECIAL, FormulaType.NUCLEAR_BINDING]:
                pass # Already in target unit
            elif config.pdg_unit == "GeV":
                pred = pred / 1000.0
            elif config.pdg_unit == "meV":
                pred = pred * 1e9
            elif config.pdg_unit == "keV":
                pred = pred * 1e6
            elif config.pdg_unit == "eV":
                pred = pred * 1e6
            
            # Check if this is an unmeasured quantity (PDG=0.0 with uncertainty=0.0)
            is_prediction = (abs(exp) < 1e-12 and config.pdg_uncertainty == 0.0)
            
            if is_prediction:
                # This is a genuine SRT prediction, not a failure
                error = 0.0  # No error to compute
                status = "PREDICT"
                status_icon = "→ PREDICT"
                predictions += 1
            elif abs(exp) < 1e-12:
                error = 0.0 if abs(pred) < 1e-9 else 100.0
                status = "PASS" if error < 1.0 else "FAIL"
                status_icon = "✓ PASS" if error < 1.0 else "✗ FAIL"
                if error < 1.0:
                    passed += 1
                else:
                    failed += 1
            else:
                error = 100 * abs(pred - exp) / exp
                status = "PASS" if error < 1.0 else "FAIL"
                status_icon = "✓ PASS" if error < 1.0 else "✗ FAIL"
                if error < 1.0:
                    passed += 1
                else:
                    failed += 1
            
            results_data.append({
                "name": name,
                "symbol": config.symbol,
                "predicted": pred,
                "experimental": exp,
                "unit": config.pdg_unit,
                "error_percent": error,
                "status": status
            })
            
            if args.verbose or status == "FAIL" or status == "PREDICT":
                print(f"{name:<15} {pred:>12.3f} {exp:>12.3f} {error:>9.4f}% {status_icon:>10}")
                
        except Exception as e:
            failed += 1
            print(f"{name:<15} {'ERROR':<12} {'-':<12} {'-':<10} {'✗ FAIL':>10}")
            if args.verbose:
                print(f"    Error: {e}")
    
    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed, {predictions} predictions out of {len(particles)}")
    print(f"Pass rate: {100*passed/(len(particles)-predictions):.1f}% (excluding predictions)")
    
    # Export results
    os.makedirs("results", exist_ok=True)
    
    # JSON Export
    with open("results/results.json", "w") as f:
        json.dump({
            "summary": {
                "total": len(particles),
                "passed": passed,
                "failed": failed,
                "pass_rate": 100*passed/len(particles)
            },
            "results": results_data
        }, f, indent=2)
    print(f"\nSaved JSON results to results/results.json")
    
    # Markdown Export
    with open("results/results.md", "w") as f:
        f.write("# SRT-Zero Validation Results\n\n")
        f.write(f"- **Total**: {len(particles)}\n")
        f.write(f"- **Passed**: {passed}\n")
        f.write(f"- **Failed**: {failed}\n")
        f.write(f"- **Pass Rate**: {100*passed/len(particles):.1f}%\n\n")
        f.write("| Particle | Symbol | Predicted | Experimental | Unit | Error (%) | Status |\n")
        f.write("|----------|--------|-----------|--------------|------|-----------|--------|\n")
        for r in results_data:
            icon = "✓" if r["status"] == "PASS" else "✗"
            f.write(f"| {r['name']} | {r['symbol']} | {r['predicted']:.6g} | {r['experimental']:.6g} | {r['unit']} | {r['error_percent']:.4f} | {icon} {r['status']} |\n")
    print(f"Saved Markdown results to results/results.md")
    
    return 0 if failed == 0 else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available particles."""
    print(f"\n{'='*60}")
    print("SRT-Zero Particle Catalog")
    print(f"{'='*60}")
    
    for ptype in ParticleType:
        particles = list_particles(ptype)
        if particles:
            print(f"\n{ptype.name}:")
            for name in particles:
                config = get_particle(name)
                print(f"  {name:<15} {config.pdg_value:>12.3f} {config.pdg_unit}")
    
    print(f"\n{len(CATALOG)} particles total")
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="srt-zero",
        description="SRT-Zero: Geometric particle mass derivation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # derive command
    derive_parser = subparsers.add_parser("derive", help="Derive a particle's mass")
    derive_parser.add_argument("particle", help="Particle name (e.g., proton, charm, B)")
    derive_parser.set_defaults(func=cmd_derive)
    
    # mine command
    mine_parser = subparsers.add_parser("mine", help="Search for mass formulas")
    mine_parser.add_argument("mass", type=float, help="Target mass in MeV")
    mine_parser.add_argument("-t", "--tolerance", type=float, default=0.1,
                            help="Tolerance percentage (default: 0.1)")
    mine_parser.set_defaults(func=cmd_mine)
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Run validation suite")
    validate_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="Show all results, not just failures")
    validate_parser.add_argument("-u", "--unique", action="store_true",
                                 help="Show only unique particles (exclude aliases)")
    validate_parser.set_defaults(func=cmd_validate)
    
    # list command
    list_parser = subparsers.add_parser("list", help="List available particles")
    list_parser.set_defaults(func=cmd_list)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
