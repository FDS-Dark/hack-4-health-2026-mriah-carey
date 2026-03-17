import { useState, useMemo, useCallback } from 'react';
import { Search } from 'lucide-react';
import './Glossary.css';

interface GlossaryTerm {
  term: string;
  definition: string;
  category: string;
}

interface GlossaryProps {
  terms: GlossaryTerm[];
}

// Common words that should NOT appear in the glossary (same as highlighting filter)
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
  'release', 'released', 'secreted', 'found', 'conducted', 'created', 'targeting',
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
  'able', 'unable', 'human', 'adult', 'tiny', 'unusual', 'chemical', 'twenty-three',
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
  'attempts', 'attempt', 'bookshelf', 'scientists', 'scientist', 'researchers', 'researcher',
  'worms', 'worm', 'blood', 'cells', 'cell', 'products', 'product', 'substances', 'substance',
  // Common adverbs
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

function Glossary({ terms }: GlossaryProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Check if a term should be shown in glossary (medical/difficult terms only)
  const shouldShowTerm = useCallback((term: GlossaryTerm): boolean => {
    const termLower = term.term.toLowerCase().trim();
    
    // Filter out very short terms
    if (termLower.length < 3) return false;
    
    // Filter out common words
    if (COMMON_WORDS.has(termLower)) return false;
    
    // Filter out numbers and units
    if (/^\d+$/.test(termLower) || /^(mg|ml|mcg|units?)$/i.test(termLower)) return false;
    
    // Always show medical categories
    const medicalCategories = ['diagnosis', 'test', 'treatment', 'anatomy'];
    if (medicalCategories.includes(term.category.toLowerCase())) return true;
    
    // Check for medical indicators
    const medicalIndicators = [
      'itis', 'osis', 'emia', 'oma', 'pathy', 'algia', 'ectomy', 'otomy',
      'scopy', 'graphy', 'gram', 'logy', 'phobia', 'philia', 'phage',
      'cyte', 'blast', 'gen', 'genic', 'trophy', 'plasia',
      'hyper', 'hypo', 'dys', 'mal', 'pseudo', 'neo', 'meta', 'para',
      'anti', 'pre', 'post', 'sub', 'super', 'trans', 'intra', 'inter',
      'peri', 'endo', 'exo', 'ecto', 'epi'
    ];
    if (medicalIndicators.some(indicator => termLower.includes(indicator))) return true;
    
    // Medical abbreviations
    if (/^[A-Z]{2,5}$/.test(term.term)) return true;
    
    // Long words (likely complex)
    if (termLower.length > 8) return true;
    
    // Multi-syllable words
    const vowelCount = (termLower.match(/[aeiouy]+/g) || []).length;
    if (vowelCount >= 3) return true;
    
    return false;
  }, []);

  // Filter and sort terms
  const filteredTerms = useMemo(() => {
    if (!terms || terms.length === 0) return [];
    
    // First filter to only medical/difficult terms
    let filtered = terms.filter(term => shouldShowTerm(term));
    
    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (term) =>
          term.term.toLowerCase().includes(query) ||
          term.definition.toLowerCase().includes(query) ||
          term.category.toLowerCase().includes(query)
      );
    }
    
    // Sort alphabetically by term
    return filtered.sort((a, b) => a.term.localeCompare(b.term));
  }, [terms, searchQuery, shouldShowTerm]);

  // Count of medical terms (after filtering)
  const medicalTermCount = useMemo(() => {
    if (!terms || terms.length === 0) return 0;
    return terms.filter(term => shouldShowTerm(term)).length;
  }, [terms, shouldShowTerm]);

  if (!terms || terms.length === 0 || medicalTermCount === 0) {
    return (
      <div className="glossary-container">
        <h2 className="glossary-title">Medical Glossary</h2>
        <p className="glossary-empty">No medical terms found in this document.</p>
      </div>
    );
  }

  return (
    <div className="glossary-container">
      <h2 className="glossary-title">Medical Glossary</h2>
      <p className="glossary-subtitle">Search and learn about medical terms from your report</p>
      
      <div className="glossary-search-container">
        <Search className="glossary-search-icon" />
        <input
          type="text"
          className="glossary-search-input"
          placeholder="Search medical terms..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="glossary-stats">
        {searchQuery ? (
          <span>Found {filteredTerms.length} term{filteredTerms.length !== 1 ? 's' : ''} matching "{searchQuery}"</span>
        ) : (
          <span>Showing {filteredTerms.length} term{filteredTerms.length !== 1 ? 's' : ''} alphabetically</span>
        )}
      </div>

      <div className="glossary-terms-list">
        {filteredTerms.length > 0 ? (
          filteredTerms.map((term, index) => (
            <div key={index} className="glossary-term-card">
              <div className="glossary-term-header">
                <h3 className="glossary-term-name">{term.term}</h3>
                <span className={`glossary-term-category category-${term.category}`}>
                  {term.category}
                </span>
              </div>
              <p className="glossary-term-definition">{term.definition}</p>
            </div>
          ))
        ) : (
          <div className="glossary-no-results">
            <p>No terms found matching "{searchQuery}"</p>
            <p className="glossary-hint">Try a different search term</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Glossary;
