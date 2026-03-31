/**
 * App-wide configuration.
 * Set API_BASE_URL to your deployed backend URL before building for distribution.
 * For local development, use your machine's LAN IP (not localhost) so devices can reach it.
 */
export const Config = {
  // Replace with your deployed API URL for production builds
  API_BASE_URL: process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000',
  API_TIMEOUT_MS: 60_000,

  // Beta feedback form (Google Form pre-fill URL)
  FEEDBACK_FORM_URL: 'https://docs.google.com/forms/d/e/YOUR_FORM_ID/viewform',

  APP_VERSION: '1.0.0-beta',
  MEDICAL_DISCLAIMER:
    'For research and educational use only. Not a medical device. ' +
    'Clinical decisions must be made by qualified healthcare professionals.',
};
