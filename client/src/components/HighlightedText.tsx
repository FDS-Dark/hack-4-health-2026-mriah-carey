import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import './HighlightedText.css';

interface GlossaryTerm {
  term: string;
  definition: string;
  category: string;
}

interface HighlightedTextProps {
  text: string;
  glossaryTerms: GlossaryTerm[];
}

interface PopupPosition {
  top: number;
  left: number;
}

// Common words that should NOT be highlighted (expanded list)
const COMMON_WORDS = new Set([
  // Articles and determiners
  'the', 'a', 'an', 'this', 'that', 'these', 'those',
  // Prepositions
  'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'into', 'onto', 'upon',
  'over', 'under', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after',
  // Common verbs
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
  'show', 'see', 'look', 'find', 'get', 'take', 'give', 'make', 'go', 'come', 'say', 'tell',
  'know', 'think', 'feel', 'want', 'need', 'use', 'work', 'call', 'try', 'ask', 'help',
  'change', 'move', 'turn', 'start', 'stop', 'continue', 'begin', 'end', 'keep', 'put', 'run',
  'set', 'seem', 'let', 'mean', 'leave', 'play', 'read', 'write', 'learn', 'live', 'believe',
  'hold', 'bring', 'happen', 'provide', 'include', 'consider', 'appear', 'create', 'expect',
  'suggest', 'allow', 'require', 'support', 'produce', 'contain', 'receive', 'remember',
  'cause', 'follow', 'develop', 'understand', 'represent', 'remain', 'involve', 'offer',
  'release', 'released', 'secreted',
  // Pronouns
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself', 'himself', 'herself',
  'itself', 'ourselves', 'themselves', 'who', 'whom', 'whose', 'which', 'what',
  // Common adjectives
  'active', 'normal', 'good', 'bad', 'big', 'small', 'large', 'little', 'high', 'low',
  'new', 'old', 'young', 'right', 'left', 'same', 'different', 'other', 'another',
  'first', 'last', 'next', 'previous', 'current', 'present', 'past', 'future',
  'long', 'short', 'full', 'empty', 'open', 'closed', 'free', 'clear', 'easy', 'hard',
  'simple', 'complex', 'common', 'rare', 'important', 'major', 'minor', 'main', 'natural',
  'possible', 'likely', 'unlikely', 'certain', 'sure', 'real', 'true', 'false', 'wrong',
  'specific', 'general', 'particular', 'various', 'similar', 'available', 'necessary',
  'able', 'unable', 'human', 'adult', 'tiny', 'unusual',
  // Common nouns
  'patient', 'doctor', 'time', 'day', 'week', 'month', 'year', 'date',
  'report', 'test', 'result', 'value', 'level', 'number', 'amount', 'size',
  'way', 'part', 'place', 'thing', 'case', 'point', 'end', 'beginning', 'middle',
  'person', 'people', 'world', 'life', 'work', 'fact', 'group', 'problem', 'hand',
  'body', 'bodies', 'area', 'system', 'program', 'question', 'government', 'company',
  'word', 'words', 'water', 'side', 'line', 'reason', 'story', 'study', 'studies',
  'example', 'family', 'member', 'head', 'eye', 'eyes', 'face', 'room', 'mother', 'father',
  'child', 'children', 'school', 'state', 'country', 'home', 'night', 'money', 'book',
  'order', 'business', 'issue', 'power', 'lot', 'hour', 'game', 'house', 'service',
  'friend', 'minute', 'idea', 'kind', 'sort', 'type', 'form', 'process', 'action',
  'effect', 'effects', 'research', 'information', 'subjects', 'target', 'priority',
  // Common adverbs - IMPORTANT: these longer words need explicit exclusion
  'very', 'more', 'most', 'less', 'least', 'also', 'only', 'just', 'even', 'still', 'yet',
  'already', 'always', 'never', 'often', 'sometimes', 'usually', 'here', 'there', 'where',
  'up', 'down', 'out', 'off', 'away', 'back', 'around', 'along', 'across',
  'however', 'therefore', 'although', 'otherwise', 'meanwhile', 'furthermore', 'moreover',
  'nevertheless', 'nonetheless', 'consequently', 'accordingly', 'subsequently', 'previously',
  'certainly', 'probably', 'possibly', 'perhaps', 'maybe', 'actually', 'basically', 'generally',
  'especially', 'specifically', 'particularly', 'primarily', 'mainly', 'mostly', 'largely',
  'simply', 'directly', 'exactly', 'nearly', 'almost', 'completely', 'entirely', 'totally',
  'absolutely', 'definitely', 'clearly', 'obviously', 'apparently', 'evidently', 'seemingly',
  'surprisingly', 'interestingly', 'importantly', 'significantly', 'typically', 'normally',
  'usually', 'frequently', 'rarely', 'occasionally', 'recently', 'currently', 'eventually',
  'finally', 'initially', 'originally', 'ultimately', 'immediately', 'suddenly', 'quickly',
  'slowly', 'carefully', 'easily', 'hardly', 'barely', 'slightly', 'highly', 'extremely',
  'relatively', 'fairly', 'quite', 'rather', 'somewhat', 'too', 'enough', 'indeed', 'instead',
  'together', 'alone', 'apart', 'forward', 'again', 'once', 'twice', 'then', 'now', 'soon',
  // Conjunctions
  'and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'as', 'than',
  'although', 'though', 'unless', 'until', 'since', 'whether', 'whereas',
  // Other common words
  'not', 'no', 'yes', 'all', 'each', 'every', 'some', 'any', 'many', 'much', 'few', 'several',
  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'both', 'either', 'neither', 'such', 'own', 'well', 'like', 'against', 'within', 'without'
]);

function HighlightedText({ text, glossaryTerms }: HighlightedTextProps) {
  const [selectedTerm, setSelectedTerm] = useState<GlossaryTerm | null>(null);
  const [popupPosition, setPopupPosition] = useState<PopupPosition | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  // Check if a term should be highlighted (medical terminology or difficult word)
  const shouldHighlightTerm = useCallback((term: GlossaryTerm): boolean => {
    const termLower = term.term.toLowerCase().trim();
    
    // Filter out very short terms (less than 3 characters)
    if (termLower.length < 3) {
      return false;
    }
    
    // Filter out common words
    if (COMMON_WORDS.has(termLower)) {
      return false;
    }
    
    // Filter out common word patterns (numbers, simple words)
    if (/^\d+$/.test(termLower) || /^(mg|ml|mcg|units?)$/i.test(termLower)) {
      return false;
    }
    
    // Always highlight if it's categorized as medical terminology
    const medicalCategories = ['diagnosis', 'test', 'treatment', 'anatomy'];
    if (medicalCategories.includes(term.category.toLowerCase())) {
      return true;
    }
    
    // Check if term contains medical indicators (common medical prefixes/suffixes)
    const medicalIndicators = [
      'itis', 'osis', 'emia', 'oma', 'pathy', 'algia', 'ectomy', 'otomy',
      'scopy', 'graphy', 'gram', 'logy', 'phobia', 'philia', 'phage',
      'cyte', 'blast', 'gen', 'genic', 'trophy', 'plasia', 'trophy',
      'hyper', 'hypo', 'dys', 'mal', 'pseudo', 'neo', 'meta', 'para',
      'anti', 'pre', 'post', 'sub', 'super', 'trans', 'intra', 'inter',
      'peri', 'endo', 'exo', 'ecto', 'epi'
    ];
    
    const hasMedicalIndicator = medicalIndicators.some(indicator => 
      termLower.includes(indicator)
    );
    
    if (hasMedicalIndicator) {
      return true;
    }
    
    // Check if term is likely a medical abbreviation (all caps, 2-5 chars)
    if (/^[A-Z]{2,5}$/.test(term.term)) {
      return true;
    }
    
    // Filter out if it's a very common English word (check word length and complexity)
    // Words longer than 8 characters are likely complex enough
    if (termLower.length > 8) {
      return true;
    }
    
    // Words with 3+ syllables (rough estimate: count vowels)
    const vowelCount = (termLower.match(/[aeiouy]+/g) || []).length;
    if (vowelCount >= 3) {
      return true;
    }
    
    // Default: don't highlight if we're not sure it's medical/difficult
    return false;
  }, []);

  // Create a map of terms for quick lookup (case-insensitive)
  const termMap = useMemo(() => {
    const map = new Map<string, GlossaryTerm>();
    glossaryTerms.forEach(term => {
      map.set(term.term.toLowerCase(), term);
    });
    return map;
  }, [glossaryTerms]);

  // Filter and sort terms by length (longest first) to match longer terms first
  const sortedTerms = useMemo(() => {
    const filtered = glossaryTerms.filter(term => shouldHighlightTerm(term));
    return filtered.sort((a, b) => b.term.length - a.term.length);
  }, [glossaryTerms, shouldHighlightTerm]);

  const handleTermClick = useCallback((e: React.MouseEvent<HTMLSpanElement>, term: GlossaryTerm) => {
    e.stopPropagation();
    
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    
    if (containerRect) {
      // Calculate position relative to container
      let top = rect.bottom - containerRect.top + 5;
      let left = rect.left - containerRect.left;
      
      // Adjust if popup would go off-screen (estimate popup width ~350px)
      const popupWidth = 350;
      const popupHeight = 150; // Estimate
      
      // Check right edge
      if (left + popupWidth > containerRect.width) {
        left = containerRect.width - popupWidth - 10;
      }
      
      // Check left edge
      if (left < 0) {
        left = 10;
      }
      
      // Check bottom edge - if popup would go below container, show above term instead
      if (top + popupHeight > containerRect.height) {
        top = rect.top - containerRect.top - popupHeight - 5;
        // If still off-screen at top, position at top of container
        if (top < 0) {
          top = 10;
        }
      }
      
      setPopupPosition({
        top: Math.max(0, top),
        left: Math.max(0, left)
      });
      setSelectedTerm(term);
    }
  }, []);

  // Highlight text with medical terms
  const highlightedText = useMemo(() => {
    if (!text || sortedTerms.length === 0) {
      return <span>{text}</span>;
    }

    const parts: Array<{ text: string; isTerm: boolean; term?: GlossaryTerm }> = [];
    let lastIndex = 0;

    // Create a copy of text to work with
    const workingText = text;
    const matches: Array<{ start: number; end: number; term: GlossaryTerm }> = [];

    // Find all matches for each term (sorted by length, longest first)
    sortedTerms.forEach(glossaryTerm => {
      const term = glossaryTerm.term;
      // Escape special regex characters and use word boundaries
      const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`\\b${escapedTerm}\\b`, 'gi');
      const textMatches: RegExpExecArray[] = [];
      let match;
      
      // Collect all matches first
      while ((match = regex.exec(workingText)) !== null) {
        textMatches.push(match);
      }

      // Add non-overlapping matches
      textMatches.forEach(match => {
        const matchStart = match.index;
        const matchEnd = match.index + match[0].length;
        
        // Check if this match overlaps with an existing match
        const overlaps = matches.some(m => 
          (matchStart < m.end && matchEnd > m.start)
        );
        
        if (!overlaps) {
          matches.push({
            start: matchStart,
            end: matchEnd,
            term: glossaryTerm
          });
        }
      });
    });

    // Sort matches by start position
    matches.sort((a, b) => a.start - b.start);

    // Build the parts array
    matches.forEach(match => {
      // Add text before the match
      if (match.start > lastIndex) {
        parts.push({
          text: workingText.substring(lastIndex, match.start),
          isTerm: false
        });
      }

      // Add the matched term
      parts.push({
        text: workingText.substring(match.start, match.end),
        isTerm: true,
        term: match.term
      });

      lastIndex = match.end;
    });

    // Add remaining text
    if (lastIndex < workingText.length) {
      parts.push({
        text: workingText.substring(lastIndex),
        isTerm: false
      });
    }

    // If no matches, return plain text
    if (parts.length === 0) {
      return <span>{text}</span>;
    }

    return (
      <>
        {parts.map((part, index) => {
          if (part.isTerm && part.term) {
            return (
              <span
                key={index}
                className="medical-term"
                onClick={(e) => handleTermClick(e, part.term!)}
              >
                {part.text}
              </span>
            );
          }
          return <span key={index}>{part.text}</span>;
        })}
      </>
    );
  }, [text, sortedTerms]);

  // Close popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(event.target as Node) &&
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setSelectedTerm(null);
        setPopupPosition(null);
      }
    };

    if (selectedTerm) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [selectedTerm]);

  return (
    <div ref={containerRef} className="highlighted-text-container">
      <div className="highlighted-text-content">
        {highlightedText}
      </div>
      {selectedTerm && popupPosition && (
        <div
          ref={popupRef}
          className="term-popup"
          style={{
            top: `${popupPosition.top}px`,
            left: `${popupPosition.left}px`
          }}
        >
          <div className="term-popup-header">
            <h4 className="term-popup-name">{selectedTerm.term}</h4>
            <span className={`term-popup-category category-${selectedTerm.category}`}>
              {selectedTerm.category}
            </span>
          </div>
          <p className="term-popup-definition">{selectedTerm.definition}</p>
          <button
            className="term-popup-close"
            onClick={() => {
              setSelectedTerm(null);
              setPopupPosition(null);
            }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

export default HighlightedText;

