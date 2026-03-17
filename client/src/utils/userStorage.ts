export interface UserData {
  name: string;
  dateOfBirth: string;
  address: string;
  preferredMedicalFacility: string;
  email?: string;
}

const STORAGE_KEY = 'benmed_user';

export const getUserData = (): UserData => {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) {
    return {
      name: 'User',
      dateOfBirth: '',
      address: '',
      preferredMedicalFacility: '',
      email: '',
    };
  }
  try {
    return JSON.parse(stored);
  } catch {
    return {
      name: 'User',
      dateOfBirth: '',
      address: '',
      preferredMedicalFacility: '',
      email: '',
    };
  }
};

export const saveUserData = (userData: Partial<UserData>): void => {
  const current = getUserData();
  const updated = { ...current, ...userData };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
};
