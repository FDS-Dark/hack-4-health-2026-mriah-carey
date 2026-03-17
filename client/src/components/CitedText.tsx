import React, { useMemo, useRef, useState, useCallback } from 'react';
import './CitedText.css';

interface GlossaryTerm {
  term: string;
  definition: string;
  category: string;
}

interface PopupPosition {
  top: number;
  left: number;
}

interface CitedTextProps {
  text: string;
  chunks: unknown[]; // Keep for API compatibility
  citations: { [sourceId: string]: string[] };
  onCitationClick?: (sourceTitle: string) => void;
  glossaryTerms?: GlossaryTerm[];
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

function CitedText({ text, citations, onCitationClick, glossaryTerms = [] }: CitedTextProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [selectedTerm, setSelectedTerm] = useState<GlossaryTerm | null>(null);
  const [popupPosition, setPopupPosition] = useState<PopupPosition | null>(null);

  // Check if a term should be highlighted
  const shouldHighlightTerm = useCallback((term: GlossaryTerm): boolean => {
    const termLower = term.term.toLowerCase().trim();
    if (termLower.length < 3) return false;
    if (COMMON_WORDS.has(termLower)) return false;
    if (/^\d+$/.test(termLower) || /^(mg|ml|mcg|units?)$/i.test(termLower)) return false;
    
    const medicalCategories = ['diagnosis', 'test', 'treatment', 'anatomy'];
    if (medicalCategories.includes(term.category.toLowerCase())) return true;
    
    const medicalIndicators = [
      'itis', 'osis', 'emia', 'oma', 'pathy', 'algia', 'ectomy', 'otomy',
      'scopy', 'graphy', 'gram', 'logy', 'phobia', 'philia', 'phage',
      'cyte', 'blast', 'gen', 'genic', 'trophy', 'plasia',
      'hyper', 'hypo', 'dys', 'mal', 'pseudo', 'neo', 'meta', 'para',
      'anti', 'pre', 'post', 'sub', 'super', 'trans', 'intra', 'inter',
      'peri', 'endo', 'exo', 'ecto', 'epi'
    ];
    if (medicalIndicators.some(indicator => termLower.includes(indicator))) return true;
    if (/^[A-Z]{2,5}$/.test(term.term)) return true;
    if (termLower.length > 8) return true;
    
    const vowelCount = (termLower.match(/[aeiouy]+/g) || []).length;
    if (vowelCount >= 3) return true;
    
    return false;
  }, []);

  // Filter glossary terms
  const sortedGlossaryTerms = useMemo(() => {
    const filtered = glossaryTerms.filter(term => shouldHighlightTerm(term));
    return filtered.sort((a, b) => b.term.length - a.term.length);
  }, [glossaryTerms, shouldHighlightTerm]);

  const handleTermClick = useCallback((e: React.MouseEvent<HTMLSpanElement>, term: GlossaryTerm) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    
    if (containerRect) {
      let top = rect.bottom - containerRect.top + 5;
      let left = rect.left - containerRect.left;
      const popupWidth = 350;
      const popupHeight = 150;
      
      if (left + popupWidth > containerRect.width) left = containerRect.width - popupWidth - 10;
      if (left < 0) left = 10;
      if (top + popupHeight > containerRect.height) {
        top = rect.top - containerRect.top - popupHeight - 5;
        if (top < 0) top = 10;
      }
      
      setPopupPosition({ top: Math.max(0, top), left: Math.max(0, left) });
      setSelectedTerm(term);
    }
  }, []);

  // Create a map of source titles to citation numbers
  const citationNumbers = useMemo(() => {
    const numbers: { [key: string]: number } = {};
    let counter = 1;
    Object.keys(citations).forEach(key => {
      numbers[key] = counter++;
    });
    return numbers;
  }, [citations]);

  // Process text to find bracketed source titles and make them clickable citations
  const processedText = useMemo(() => {
    let processed = text;
    
    // Find all bracketed text that matches source titles: [Source Title Here]
    // These are inline citations that Gemini puts in the text
    const sourceTitles = Object.keys(citations);
    
    // Sort by length (longest first) to avoid partial matches
    const sortedTitles = [...sourceTitles].sort((a, b) => b.length - a.length);
    
    sortedTitles.forEach((sourceTitle) => {
      const citationNum = citationNumbers[sourceTitle] || 0;
      
      // Look for [Source Title] pattern in text
      const escapedTitle = sourceTitle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const bracketPattern = new RegExp(`\\[${escapedTitle}\\]`, 'gi');
      
      processed = processed.replace(bracketPattern, (match) => {
        return `%%CITE_START_${citationNum}_${sourceTitle}%%${match}%%CITE_END%%`;
      });
    });
    
    return processed;
  }, [text, citations, citationNumbers]);

  // Handle citation click
  const handleCitationClick = (sourceTitle: string) => {
    if (onCitationClick) {
      onCitationClick(sourceTitle);
    }
  };

  // Render markdown text with citations (highlighted and clickable)
  const renderMarkdown = (content: string): React.ReactNode[] => {
    const elements: React.ReactNode[] = [];
    let keyCounter = 0;
    
    // Process the content to find citation blocks
    let remaining = content;
    
    while (remaining.length > 0) {
      // Look for the start of a citation block
      const startMatch = remaining.match(/%%CITE_START_(\d+)_([^%]+)%%/);
      
      if (startMatch && startMatch.index !== undefined) {
        // Add text before the citation
        if (startMatch.index > 0) {
          const textBefore = remaining.substring(0, startMatch.index);
          elements.push(
            <span key={`text-${keyCounter++}`}>
              {parseInlineMarkdownWithGlossary(textBefore)}
            </span>
          );
        }
        
        const citationNum = startMatch[1];
        const sourceTitle = startMatch[2];
        
        // Find the end marker
        const afterStart = remaining.substring(startMatch.index + startMatch[0].length);
        const endIndex = afterStart.indexOf('%%CITE_END%%');
        
        if (endIndex !== -1) {
          // Create the highlighted, clickable citation element
          // Display as a clean superscript number instead of the full bracketed title
          elements.push(
            <sup
              key={`citation-${keyCounter++}`}
              className="cited-text-link"
              onClick={() => handleCitationClick(sourceTitle)}
              title={`Source: ${sourceTitle}`}
            >
              [{citationNum}]
            </sup>
          );
          
          // Continue with the rest
          remaining = afterStart.substring(endIndex + '%%CITE_END%%'.length);
        } else {
          // No end marker found, treat as regular text
          remaining = remaining.substring(startMatch.index + startMatch[0].length);
        }
      } else {
        // No more citations, add remaining text
        elements.push(
          <span key={`text-${keyCounter++}`}>
            {parseInlineMarkdownWithGlossary(remaining)}
          </span>
        );
        break;
      }
    }
    
    return elements;
  };

  // Parse inline markdown (bold, italic) and return as React nodes with glossary highlights
  const parseInlineMarkdownWithGlossary = (text: string): React.ReactNode => {
    // First apply markdown formatting
    let result = text;
    result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    result = result.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    result = result.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
    result = result.replace(/(?<!_)_([^_]+)_(?!_)/g, '<em>$1</em>');
    
    // If no glossary terms, return as dangerouslySetInnerHTML
    if (sortedGlossaryTerms.length === 0) {
      return <span dangerouslySetInnerHTML={{ __html: result }} />;
    }
    
    // Find glossary term matches
    const matches: Array<{ start: number; end: number; term: GlossaryTerm; matchedText: string }> = [];
    
    sortedGlossaryTerms.forEach(glossaryTerm => {
      const escapedTerm = glossaryTerm.term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`\\b${escapedTerm}\\b`, 'gi');
      let match;
      
      while ((match = regex.exec(result)) !== null) {
        const matchStart = match.index;
        const matchEnd = match.index + match[0].length;
        
        // Check for overlap
        const overlaps = matches.some(m => (matchStart < m.end && matchEnd > m.start));
        
        if (!overlaps) {
          matches.push({
            start: matchStart,
            end: matchEnd,
            term: glossaryTerm,
            matchedText: match[0]
          });
        }
      }
    });
    
    if (matches.length === 0) {
      return <span dangerouslySetInnerHTML={{ __html: result }} />;
    }
    
    // Sort matches by position
    matches.sort((a, b) => a.start - b.start);
    
    // Build parts array
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    
    matches.forEach((match, idx) => {
      if (match.start > lastIndex) {
        parts.push(
          <span key={`text-${idx}`} dangerouslySetInnerHTML={{ __html: result.substring(lastIndex, match.start) }} />
        );
      }
      
      parts.push(
        <span
          key={`term-${idx}`}
          className="medical-term"
          onClick={(e) => handleTermClick(e, match.term)}
        >
          {match.matchedText}
        </span>
      );
      
      lastIndex = match.end;
    });
    
    if (lastIndex < result.length) {
      parts.push(
        <span key="text-end" dangerouslySetInnerHTML={{ __html: result.substring(lastIndex) }} />
      );
    }
    
    return <>{parts}</>;
  };

  // Render the full content with proper markdown structure
  const renderContent = () => {
    const lines = processedText.split('\n');
    const elements: React.ReactNode[] = [];
    let keyCounter = 0;
    let currentList: React.ReactNode[] = [];
    let inList = false;

    const flushList = () => {
      if (currentList.length > 0) {
        elements.push(
          <ul key={`list-${keyCounter++}`} className="markdown-list">
            {currentList}
          </ul>
        );
        currentList = [];
      }
      inList = false;
    };

    lines.forEach((line) => {
      const trimmedLine = line.trim();
      
      // Skip empty lines but flush list
      if (!trimmedLine) {
        flushList();
        return;
      }

      // Handle headers
      const headerMatch = trimmedLine.match(/^(#{1,6})\s+(.+)$/);
      if (headerMatch) {
        flushList();
        const level = headerMatch[1].length;
        const headerText = headerMatch[2];
        const headerElement = React.createElement(
          `h${level}`,
          { key: `header-${keyCounter++}`, className: `markdown-h${level}` },
          renderMarkdown(headerText)
        );
        elements.push(headerElement);
        return;
      }

      // Handle list items (*, -, or numbered)
      const listMatch = trimmedLine.match(/^[\*\-]\s+(.+)$/);
      if (listMatch) {
        inList = true;
        currentList.push(
          <li key={`li-${keyCounter++}`}>
            {renderMarkdown(listMatch[1])}
          </li>
        );
        return;
      }

      // If we were in a list but this line isn't a list item, flush the list
      if (inList) {
        flushList();
      }

      // Regular paragraph
      elements.push(
        <p key={`p-${keyCounter++}`} className="markdown-paragraph">
          {renderMarkdown(trimmedLine)}
        </p>
      );
    });

    // Flush any remaining list
    flushList();

    return elements;
  };

  return (
    <div className="cited-text-container" ref={containerRef}>
      {renderContent()}
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

export default CitedText;
