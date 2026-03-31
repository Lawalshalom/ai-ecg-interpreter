export type RiskLevel = 'HIGH' | 'MODERATE' | 'LOW';
export type DiagnosisLevel = 'HIGH' | 'MODERATE' | 'POSSIBLE';

export interface ECGFeatures {
  heart_rate?: number;
  qrs_duration?: number;
  qtc_interval?: number;
  pr_interval?: number;
  age?: number;
  sex?: string;
}

export interface Diagnosis {
  name: string;
  confidence: number;
  level: DiagnosisLevel;
  description: string;
  findings: string[];
  source: string;
}

export interface PTBXLResult {
  probabilities: number[];
  labels: string[];
}

export interface WaveformMeta {
  leads_detected: number;
  source: string;
}

export interface AnalysisResult {
  success: boolean;
  risk_level: RiskLevel;
  shd_probability: number;
  ecg_features: ECGFeatures;
  probabilities: number[];   // 12 EchoNext condition probabilities
  labels: string[];          // 12 EchoNext condition names
  diagnoses: Diagnosis[];
  ptbxl_result?: PTBXLResult;
  waveform: number[][];      // 12 × 500 (downsampled for display)
  lead_names: string[];
  waveform_meta: WaveformMeta;
}

export interface PatientInfo {
  age?: number;
  sex?: 'Male' | 'Female';
}

export interface UploadState {
  uri: string;
  name: string;
  mimeType: string;
  bytes?: Uint8Array;
}
