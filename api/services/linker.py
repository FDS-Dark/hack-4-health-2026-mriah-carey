"""
Linker Service: Validates that all input facts are preserved in the output.

Uses proposition-level source <-> summary alignment to:
1. Extract atomic facts from both source and summary
2. Align and match propositions deterministically
3. Report coverage, hallucinations, and evidence links

Synonym matching is powered by:
- MeSH (Medical Subject Headings) for standardized medical terminology
- Fallback synonyms for common clinical report phrases not in MeSH
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
from google import genai

try:
    from .mesh_synonyms import get_hybrid_index, HybridSynonymIndex
except ImportError:
    from mesh_synonyms import get_hybrid_index, HybridSynonymIndex

load_dotenv()

logger = logging.getLogger(__name__)

# Similarity thresholds for proposition matching
EXACT_MATCH_THRESHOLD = 0.8
SEMANTIC_MATCH_THRESHOLD = 0.5
DEFAULT_ALIGNMENT_THRESHOLD = 0.3

# Number word mappings for normalization
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20"
}
NUMBER_DIGITS = {v: k for k, v in NUMBER_WORDS.items()}

EXTRACTION_PROMPT = """You are an expert medical information extractor. Extract ALL atomic facts/propositions from this medical text.

For each fact, provide:
- text: the exact fact as stated (brief, one sentence max)
- category: one of "finding", "measurement", "anatomy", "condition", "recommendation", "procedure", "negation", "patient_info", "metadata"
- assertion: one of "present", "absent", "possible", "historical"
- normalized: a normalized canonical form (lowercase, standard medical terminology)

Be exhaustive - extract EVERY piece of medical information including:
- All findings (positive and negative)
- All measurements and values
- All anatomical locations mentioned
- All conditions and diagnoses
- All recommendations
- Patient demographics mentioned

Return a JSON array of objects with these fields.

Medical Text:
{text}"""


@dataclass
class Proposition:
    """A single atomic fact extracted from text."""
    id: str
    text: str
    category: str  # e.g., "finding", "measurement", "recommendation", "negation"
    assertion: str  # "present", "absent", "possible", "historical"
    normalized: str  # normalized/canonical form for matching
    span_start: Optional[int] = None
    span_end: Optional[int] = None


@dataclass
class Alignment:
    """Alignment between a summary proposition and source proposition(s)."""
    summary_prop: Proposition
    source_ids: list[str] = field(default_factory=list)
    match_type: str = "none"  # "exact", "semantic", "partial", "none"
    confidence: float = 0.0


@dataclass
class LinkageReport:
    """Complete report of source-summary alignment."""
    coverage_score: float
    hallucination_rate: float
    total_source_props: int
    total_summary_props: int
    matched_source_props: int
    unmatched_source_props: int
    unsupported_summary_props: int
    assertion_mismatches: list[dict]
    alignments: list[dict]
    source_propositions: list[dict]
    summary_propositions: list[dict]
    missing_facts: list[dict]
    potential_hallucinations: list[dict]


class LinkerService:
    def __init__(self, load_mesh: bool = True, include_mesh_supplemental: bool = False):
        """
        Initialize the LinkerService.
        
        Args:
            load_mesh: Whether to load MeSH synonyms (can be slow on first load)
            include_mesh_supplemental: Whether to include MeSH supplemental concepts
        """
        self.client = genai.Client()
        
        # Load the hybrid synonym index (MeSH + fallback)
        logger.info("Loading synonym index...")
        self._synonym_index: HybridSynonymIndex = get_hybrid_index(
            load_mesh=load_mesh,
            include_supplemental=include_mesh_supplemental
        )
        logger.info("Synonym index loaded.")

    def extract_propositions(self, text: str, prefix: str = "prop") -> list[Proposition]:
        """Extract atomic propositions from text using Gemini."""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=EXTRACTION_PROMPT.format(text=text),
            config={"response_mime_type": "application/json"}
        )
        
        response_text = response.text or ""
        try:
            props_data = json.loads(response_text)
            if not isinstance(props_data, list):
                props_data = [props_data]
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                props_data = json.loads(match.group())
            else:
                props_data = []
        
        propositions = []
        for i, p in enumerate(props_data):
            prop = Proposition(
                id=f"{prefix}_{i+1}",
                text=p.get("text", ""),
                category=p.get("category", "finding"),
                assertion=p.get("assertion", "present"),
                normalized=p.get("normalized", p.get("text", "").lower())
            )
            propositions.append(prop)
        
        return propositions

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation except hyphens
        text = re.sub(r'[^\w\s\-]', '', text)
        return text

    def normalize_numbers(self, text: str) -> str:
        """Convert number words to digits for consistent matching."""
        text = text.lower()
        words = text.split()
        normalized = []
        for word in words:
            # Convert word numbers to digits
            if word in NUMBER_WORDS:
                normalized.append(NUMBER_WORDS[word])
            else:
                normalized.append(word)
        return ' '.join(normalized)

    def get_synonyms(self, word: str) -> set[str]:
        """Get all synonyms for a word, including the word itself."""
        word_lower = word.lower()
        synonyms = self._synonym_index.get_synonyms(word_lower)
        if synonyms:
            return synonyms
        return {word_lower}

    def get_canonical(self, term: str) -> str:
        """Get the canonical form of a term, or the term itself if not found."""
        canonical = self._synonym_index.get_canonical(term)
        return canonical if canonical else term.lower()

    def expand_with_synonyms(self, words: set[str]) -> set[str]:
        """Expand a set of words to include all their synonyms."""
        expanded = set()
        for word in words:
            expanded.add(word)
            expanded.update(self.get_synonyms(word))
        return expanded

    def canonicalize_text(self, text: str) -> str:
        """
        Canonicalize text by replacing known terms with their canonical forms.
        
        Uses MeSH and fallback synonyms for standardization.
        """
        return self._synonym_index.canonicalize(text)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute word overlap similarity with synonym and number normalization.
        
        Uses MeSH-backed synonyms and canonicalization for improved matching.
        """
        # Normalize text and numbers
        normalized1 = self.normalize_numbers(self.normalize_text(text1))
        normalized2 = self.normalize_numbers(self.normalize_text(text2))
        
        # Also try canonicalized versions for matching
        canonical1 = self.normalize_numbers(self.normalize_text(self.canonicalize_text(text1)))
        canonical2 = self.normalize_numbers(self.normalize_text(self.canonicalize_text(text2)))
        
        # Get word sets from both normalized and canonicalized versions
        words1 = set(normalized1.split()) | set(canonical1.split())
        words2 = set(normalized2.split()) | set(canonical2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Compute matches: a word matches if it or any synonym is in the other set
        matches = 0
        for word in words1:
            word_synonyms = self.get_synonyms(word)
            if word_synonyms & words2:  # If any synonym is in words2
                matches += 1
            elif word in words2:
                matches += 1
        
        for word in words2:
            word_synonyms = self.get_synonyms(word)
            if word_synonyms & words1:  # If any synonym is in words1
                matches += 1
            elif word in words1:
                matches += 1
        
        # Dice coefficient (more forgiving than Jaccard for different length texts)
        total_words = len(words1) + len(words2)
        return matches / total_words if total_words > 0 else 0.0

    def align_propositions(
        self,
        source_props: list[Proposition],
        summary_props: list[Proposition],
        threshold: float = DEFAULT_ALIGNMENT_THRESHOLD
    ) -> list[Alignment]:
        """Align summary propositions to source propositions."""
        alignments = []

        for summary_prop in summary_props:
            best_matches = self._find_matches(summary_prop, source_props, threshold)
            alignment = self._create_alignment(summary_prop, best_matches)
            alignments.append(alignment)

        return alignments

    def _find_matches(
        self,
        summary_prop: Proposition,
        source_props: list[Proposition],
        threshold: float
    ) -> list[tuple[str, float, Proposition]]:
        """Find source propositions that match a summary proposition."""
        matches = []

        for source_prop in source_props:
            sim_normalized = self.compute_similarity(
                summary_prop.normalized,
                source_prop.normalized
            )
            sim_text = self.compute_similarity(
                summary_prop.text,
                source_prop.text
            )
            similarity = max(sim_normalized, sim_text)

            if similarity >= threshold:
                matches.append((source_prop.id, similarity, source_prop))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _create_alignment(
        self,
        summary_prop: Proposition,
        matches: list[tuple[str, float, Proposition]]
    ) -> Alignment:
        """Create an Alignment object from matches."""
        if not matches:
            return Alignment(
                summary_prop=summary_prop,
                source_ids=[],
                match_type="none",
                confidence=0.0
            )

        top_confidence = matches[0][1]
        if top_confidence > EXACT_MATCH_THRESHOLD:
            match_type = "exact"
        elif top_confidence > SEMANTIC_MATCH_THRESHOLD:
            match_type = "semantic"
        else:
            match_type = "partial"

        return Alignment(
            summary_prop=summary_prop,
            source_ids=[m[0] for m in matches[:3]],
            match_type=match_type,
            confidence=top_confidence
        )

    def check_assertion_consistency(
        self,
        source_props: list[Proposition],
        alignments: list[Alignment]
    ) -> list[dict]:
        """Check if assertion types match between aligned propositions."""
        source_by_id = {p.id: p for p in source_props}
        mismatches = []

        for alignment in alignments:
            if not (alignment.source_ids and alignment.confidence > SEMANTIC_MATCH_THRESHOLD):
                continue

            source_prop = source_by_id.get(alignment.source_ids[0])
            if not source_prop:
                continue

            if alignment.summary_prop.assertion != source_prop.assertion:
                mismatches.append({
                    "summary_fact_id": alignment.summary_prop.id,
                    "summary_text": alignment.summary_prop.text,
                    "source_fact_id": source_prop.id,
                    "source_text": source_prop.text,
                    "source_assertion": source_prop.assertion,
                    "summary_assertion": alignment.summary_prop.assertion
                })

        return mismatches

    def link(self, source_text: str, summary_text: str) -> LinkageReport:
        """
        Perform full linkage analysis between source and summary.

        Args:
            source_text: The original medical report text
            summary_text: The simplified/summarized text

        Returns:
            LinkageReport with coverage, hallucination rates, and alignments
        """
        source_props = self.extract_propositions(source_text, prefix="src")
        summary_props = self.extract_propositions(summary_text, prefix="sum")
        alignments = self.align_propositions(source_props, summary_props)
        assertion_mismatches = self.check_assertion_consistency(source_props, alignments)

        return self._build_report(
            source_props, summary_props, alignments, assertion_mismatches
        )

    def _build_report(
        self,
        source_props: list[Proposition],
        summary_props: list[Proposition],
        alignments: list[Alignment],
        assertion_mismatches: list[dict]
    ) -> LinkageReport:
        """Build the final LinkageReport from analysis results."""
        matched_source_ids = set()
        for alignment in alignments:
            matched_source_ids.update(alignment.source_ids)

        total_source = len(source_props)
        total_summary = len(summary_props)
        matched_count = len(matched_source_ids)

        coverage = matched_count / total_source if total_source > 0 else 1.0

        unsupported = [a for a in alignments if not a.source_ids]
        hallucination_rate = len(unsupported) / total_summary if total_summary > 0 else 0.0

        all_source_ids = {p.id for p in source_props}
        missing_ids = all_source_ids - matched_source_ids
        missing_facts = [asdict(p) for p in source_props if p.id in missing_ids]

        potential_hallucinations = [
            {"id": a.summary_prop.id, "text": a.summary_prop.text, "category": a.summary_prop.category}
            for a in unsupported
        ]

        alignment_dicts = [
            {
                "summary_fact": a.summary_prop.text,
                "summary_id": a.summary_prop.id,
                "source_ids": a.source_ids,
                "match_type": a.match_type,
                "confidence": round(a.confidence, 4)
            }
            for a in alignments
        ]

        return LinkageReport(
            coverage_score=round(coverage, 4),
            hallucination_rate=round(hallucination_rate, 4),
            total_source_props=total_source,
            total_summary_props=total_summary,
            matched_source_props=matched_count,
            unmatched_source_props=len(missing_ids),
            unsupported_summary_props=len(unsupported),
            assertion_mismatches=assertion_mismatches,
            alignments=alignment_dicts,
            source_propositions=[asdict(p) for p in source_props],
            summary_propositions=[asdict(p) for p in summary_props],
            missing_facts=missing_facts,
            potential_hallucinations=potential_hallucinations
        )

    def print_report(self, report: LinkageReport):
        """Pretty print the linkage report."""
        print("\n" + "=" * 70)
        print("📊 LINKAGE ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n📈 METRICS:")
        print(f"   Coverage Score:      {report.coverage_score:.1%}")
        print(f"   Hallucination Rate:  {report.hallucination_rate:.1%}")
        print(f"   Source Facts:        {report.total_source_props}")
        print(f"   Summary Facts:       {report.total_summary_props}")
        print(f"   Matched:             {report.matched_source_props}")
        print(f"   Missing:             {report.unmatched_source_props}")
        print(f"   Unsupported:         {report.unsupported_summary_props}")
        
        if report.assertion_mismatches:
            print(f"\n⚠️  ASSERTION MISMATCHES ({len(report.assertion_mismatches)}):")
            for m in report.assertion_mismatches:
                print(f"   • Source [{m['source_assertion']}]: \"{m['source_text'][:50]}...\"")
                print(f"     Summary [{m['summary_assertion']}]: \"{m['summary_text'][:50]}...\"")
        
        if report.missing_facts:
            print(f"\n❌ MISSING FACTS ({len(report.missing_facts)}):")
            for fact in report.missing_facts[:10]:  # Show first 10
                print(f"   • [{fact['category']}] {fact['text'][:70]}...")
            if len(report.missing_facts) > 10:
                print(f"   ... and {len(report.missing_facts) - 10} more")
        
        if report.potential_hallucinations:
            print(f"\n🚨 POTENTIAL HALLUCINATIONS ({len(report.potential_hallucinations)}):")
            for h in report.potential_hallucinations:
                print(f"   • [{h['category']}] {h['text'][:70]}...")
        
        print(f"\n✅ ALIGNED FACTS (showing top matches):")
        high_confidence = [a for a in report.alignments if a['confidence'] > 0.5]
        for a in high_confidence[:10]:  # Show first 10 high-confidence
            print(f"   • \"{a['summary_fact'][:50]}...\"")
            print(f"     ↳ Sources: {a['source_ids'][:3]} ({a['match_type']}, {a['confidence']:.0%})")
        
        if len(high_confidence) > 10:
            print(f"   ... and {len(high_confidence) - 10} more aligned facts")
        
        print("\n" + "=" * 70)

    def to_json(self, report: LinkageReport) -> str:
        """Convert report to JSON string."""
        return json.dumps(asdict(report), indent=2)

    def write_report(self, report: LinkageReport, output_path: str):
        """Write the linkage report to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)