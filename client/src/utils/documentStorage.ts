export interface DocumentData {
  id: string;
  date: string;
  fileName: string;
  summary: string;
  originalDiagnosis: string;
  simplifiedDiagnosis: string;
}

const STORAGE_KEY = 'benmed_documents';

export const saveDocument = (document: Omit<DocumentData, 'id' | 'date'>): DocumentData => {
  const documents = getDocuments();
  const newDocument: DocumentData = {
    ...document,
    id: Date.now().toString(),
    date: new Date().toISOString(),
  };
  documents.unshift(newDocument); // Add to beginning
  localStorage.setItem(STORAGE_KEY, JSON.stringify(documents));
  return newDocument;
};

export const getDocuments = (): DocumentData[] => {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch {
    return [];
  }
};

export const getDocumentById = (id: string): DocumentData | null => {
  const documents = getDocuments();
  return documents.find(doc => doc.id === id) || null;
};

export const deleteDocument = (id: string): void => {
  const documents = getDocuments();
  const filtered = documents.filter(doc => doc.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const formatDateShort = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};
