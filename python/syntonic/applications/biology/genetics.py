"""
Genetics - DNA structure and genetic code from SRT.

DNA is crystallized Tv history.
Schrödinger's "aperiodic crystal" is literally true:
DNA stores the accumulated Tv phase history as base sequence.

The genetic code (64 codons, 20 amino acids) emerges from T⁴ topology.
"""

from __future__ import annotations

from typing import Dict, Tuple

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC


class DNAStructure:
    """
    DNA as crystallized Tv history.

    Schrödinger's "aperiodic crystal" is literally true:
    DNA stores the accumulated Tv phase history as base sequence.

    Structural features with golden ratio connections:
    - Major/minor groove ratio: 22/12 ≈ φ(1 + 8q)
    - α-helix: 3.6 residues/turn ≈ 2φ + 13q

    Example:
        >>> dna = DNAStructure()
        >>> dna.groove_ratio()
        1.833...
        >>> dna.helix_residues_per_turn()
        3.59...
    """

    # DNA structural parameters
    MAJOR_GROOVE = 22  # Angstroms (approximate)
    MINOR_GROOVE = 12  # Angstroms (approximate)
    BASE_PAIRS_PER_TURN = 10.5
    RISE_PER_BASE = 3.4  # Angstroms

    def groove_ratio(self) -> float:
        """
        Major/Minor groove ratio.

        22/12 ≈ φ(1 + 8q) ≈ 1.83

        The golden ratio appears in DNA geometry!

        Returns:
            Groove ratio
        """
        return self.MAJOR_GROOVE / self.MINOR_GROOVE

    def groove_ratio_predicted(self) -> float:
        """
        SRT prediction for groove ratio.

        φ × (1 + 8q) ≈ 1.83

        Returns:
            Predicted ratio
        """
        return PHI_NUMERIC * (1 + 8 * Q_DEFICIT_NUMERIC)

    def helix_residues_per_turn(self) -> float:
        """
        α-helix: residues per turn.

        2φ + 13q ≈ 3.59

        Experiment: 3.6 → 0.2% agreement

        Returns:
            Residues per turn
        """
        return 2 * PHI_NUMERIC + 13 * Q_DEFICIT_NUMERIC

    def pitch(self) -> float:
        """
        DNA helix pitch (rise per full turn).

        Returns:
            Pitch in Angstroms
        """
        return self.BASE_PAIRS_PER_TURN * self.RISE_PER_BASE

    def schrodinger_aperiodic_crystal(self) -> str:
        """
        Explain Schrödinger's insight in SRT terms.

        Returns:
            Explanation
        """
        return """
Schrödinger's "Aperiodic Crystal" in SRT:

In 1944, Schrödinger predicted that genetic material must be an
"aperiodic crystal" - ordered but not repetitive.

SRT explanation:
- DNA IS an aperiodic crystal
- It stores the accumulated Tv phase history
- Each base pair represents a unit of Tv information
- The sequence is not periodic because history is not periodic
- But it has crystalline order because it follows winding rules

The double helix structure:
- Two complementary strands (information redundancy)
- Base pairing (A-T, G-C) from winding compatibility
- Sugar-phosphate backbone (structural scaffold)
- Right-handed helix (chirality preference)

Why RIGHT-handed?
- Homochirality is required for strong Tv hooks
- Left-handed DNA exists but is rare
- The choice was made early and locked in
- All life descends from that choice
"""

    def information_density(self) -> str:
        """
        Explain DNA information density.

        Returns:
            Description
        """
        return """
DNA Information Density:

Each base pair: 2 bits (4 possibilities: A, T, G, C)
Per turn (10.5 bp): ~21 bits
Per nanometer: ~6 bits
Per human genome (3 × 10^9 bp): ~6 × 10^9 bits = 750 MB

But with compression (much is repetitive/unused):
Effective information: ~200-400 MB

This is REMARKABLY efficient:
- More dense than any human technology
- Self-replicating with error correction
- Stable for billions of years (with repair)

Why this specific density?
- It maximizes information per unit syntony cost
- More dense → more interference
- Less dense → inefficient use of chemistry
- Golden ratio optimization
"""


class GeneticCode:
    """
    64 codons, 20 amino acids from T⁴ representation theory.

    64 = 4³ (3 bases from 4 nucleotides)
    20 = number of standard amino acids

    The redundancy pattern follows winding symmetry.

    Example:
        >>> gc = GeneticCode()
        >>> gc.codon_count()
        64
        >>> gc.amino_acid_count()
        20
    """

    # Standard genetic code
    CODONS_TO_AMINO = {
        "UUU": "Phe",
        "UUC": "Phe",
        "UUA": "Leu",
        "UUG": "Leu",
        "UCU": "Ser",
        "UCC": "Ser",
        "UCA": "Ser",
        "UCG": "Ser",
        "UAU": "Tyr",
        "UAC": "Tyr",
        "UAA": "Stop",
        "UAG": "Stop",
        "UGU": "Cys",
        "UGC": "Cys",
        "UGA": "Stop",
        "UGG": "Trp",
        "CUU": "Leu",
        "CUC": "Leu",
        "CUA": "Leu",
        "CUG": "Leu",
        "CCU": "Pro",
        "CCC": "Pro",
        "CCA": "Pro",
        "CCG": "Pro",
        "CAU": "His",
        "CAC": "His",
        "CAA": "Gln",
        "CAG": "Gln",
        "CGU": "Arg",
        "CGC": "Arg",
        "CGA": "Arg",
        "CGG": "Arg",
        "AUU": "Ile",
        "AUC": "Ile",
        "AUA": "Ile",
        "AUG": "Met",
        "ACU": "Thr",
        "ACC": "Thr",
        "ACA": "Thr",
        "ACG": "Thr",
        "AAU": "Asn",
        "AAC": "Asn",
        "AAA": "Lys",
        "AAG": "Lys",
        "AGU": "Ser",
        "AGC": "Ser",
        "AGA": "Arg",
        "AGG": "Arg",
        "GUU": "Val",
        "GUC": "Val",
        "GUA": "Val",
        "GUG": "Val",
        "GCU": "Ala",
        "GCC": "Ala",
        "GCA": "Ala",
        "GCG": "Ala",
        "GAU": "Asp",
        "GAC": "Asp",
        "GAA": "Glu",
        "GAG": "Glu",
        "GGU": "Gly",
        "GGC": "Gly",
        "GGA": "Gly",
        "GGG": "Gly",
    }

    # Nucleotide to T⁴ direction mapping
    BASE_TO_WINDING = {"U": 0, "A": 1, "G": 2, "C": 3}

    def codon_count(self) -> int:
        """
        Total number of codons.

        64 = 4³

        Returns:
            Number of codons
        """
        return 64

    def amino_acid_count(self) -> int:
        """
        Number of standard amino acids.

        20 (plus 3 stop codons)

        Returns:
            Number of amino acids
        """
        return 20

    def redundancy(self) -> Dict[str, int]:
        """
        Codon redundancy (degeneracy) for each amino acid.

        Returns:
            Dict mapping amino acid to number of codons
        """
        counts: Dict[str, int] = {}
        for codon, aa in self.CODONS_TO_AMINO.items():
            counts[aa] = counts.get(aa, 0) + 1
        return counts

    def translate(self, codon: str) -> str:
        """
        Translate a codon to amino acid.

        Args:
            codon: Three-letter RNA codon (e.g., 'AUG')

        Returns:
            Amino acid name or 'Stop'
        """
        return self.CODONS_TO_AMINO.get(codon.upper(), "Unknown")

    def codon_to_winding(self, codon: str) -> Tuple[int, int, int, int]:
        """
        Map genetic codon to T⁴ winding configuration.

        Each nucleotide (A, U, G, C) maps to a direction on T⁴.

        Args:
            codon: Three-letter codon

        Returns:
            4-tuple (n7, n8, n9, n10)
        """
        if len(codon) != 3:
            raise ValueError("Codon must be 3 bases")

        codon = codon.upper().replace("T", "U")  # DNA to RNA
        n7 = self.BASE_TO_WINDING[codon[0]]
        n8 = self.BASE_TO_WINDING[codon[1]]
        n9 = self.BASE_TO_WINDING[codon[2]]
        n10 = 0  # Fourth dimension for codon context

        return (n7, n8, n9, n10)

    def why_20_amino_acids(self) -> str:
        """
        Explain why life uses exactly 20 amino acids.

        Returns:
            Explanation
        """
        return """
Why 20 Amino Acids?

The number 20 is NOT arbitrary:

Mathematical structure:
- 64 codons = 4³
- 64 = 20 + 20 + 20 + 4 (with redundancy pattern)
- Or: 64 = 20 × 3 + 3 stop + 1 start (Met)

SRT interpretation:
- 20 corresponds to icosahedral symmetry
- Icosahedron has 20 faces
- Related to golden ratio: vertices at (0, ±1, ±φ)
- The amino acid set reflects T⁴ → M⁴ projection

The redundancy pattern is optimized:
- Wobble position (3rd base) often doesn't matter
- Similar codons → similar amino acids
- Minimizes translation errors
- Follows winding distance on T⁴

Why not more?
- 20 is sufficient for all protein structures
- More would increase error rates
- The code is frozen (hard to change)

Why not fewer?
- Need hydrophobic/hydrophilic variety
- Need positive/negative charges
- Need small/large residues
- 20 is optimal for folding diversity
"""

    def universality(self) -> str:
        """
        Explain the universality of the genetic code.

        Returns:
            Explanation
        """
        return """
Universality of the Genetic Code:

The SAME code is used by virtually ALL life:
- Bacteria, archaea, eukaryotes
- Plants, animals, fungi
- Minor variations exist but core is identical

This proves:
1. All life shares a common ancestor
2. The code was established VERY early
3. It became "frozen" - too costly to change

SRT interpretation:
- The code reflects optimal T⁴ → M⁴ projection
- It's not arbitrary but geometrically constrained
- Any life in the universe would likely converge
  to similar code structure

Small variations:
- Mitochondria use slightly different code
- Some organisms reassign stop codons
- But the basic structure is universal
"""

    def __repr__(self) -> str:
        return "GeneticCode(codons=64, amino_acids=20)"
