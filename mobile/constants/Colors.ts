export const Colors = {
  primary: '#1a237e',
  primaryLight: '#3949ab',
  primaryDark: '#0d1257',

  danger: '#c62828',
  dangerLight: '#ef5350',
  dangerBg: '#ffebee',

  warning: '#e65100',
  warningLight: '#ff8f00',
  warningBg: '#fff3e0',

  success: '#2e7d32',
  successLight: '#43a047',
  successBg: '#e8f5e9',

  background: '#f5f7fa',
  surface: '#ffffff',
  surfaceVariant: '#eef0f8',
  border: '#dde1ee',

  textPrimary: '#1a1a2e',
  textSecondary: '#5c6080',
  textDisabled: '#9e9eb8',

  ecgTrace: '#c62828',
  ecgGrid: '#ffcdd2',
  ecgBackground: '#fff9f9',

  white: '#ffffff',
  black: '#000000',
};

export const RiskColors = {
  HIGH: {
    bg: Colors.dangerBg,
    border: Colors.dangerLight,
    text: Colors.danger,
    badge: Colors.danger,
  },
  MODERATE: {
    bg: Colors.warningBg,
    border: Colors.warningLight,
    text: Colors.warning,
    badge: Colors.warning,
  },
  LOW: {
    bg: Colors.successBg,
    border: Colors.successLight,
    text: Colors.success,
    badge: Colors.success,
  },
};
