import axios, { AxiosError } from 'axios';
import { Config } from '@/constants/Config';
import type { AnalysisResult, PatientInfo } from '@/types';

const client = axios.create({
  baseURL: Config.API_BASE_URL,
  timeout: Config.API_TIMEOUT_MS,
});

export async function analyzeECG(
  fileUri: string,
  fileName: string,
  mimeType: string,
  patient: PatientInfo
): Promise<AnalysisResult> {
  const form = new FormData();

  // React Native FormData accepts this object shape
  form.append('file', {
    uri: fileUri,
    name: fileName,
    type: mimeType,
  } as unknown as Blob);

  if (patient.age !== undefined) {
    form.append('age', String(patient.age));
  }
  if (patient.sex) {
    form.append('sex', patient.sex);
  }

  try {
    const { data } = await client.post<AnalysisResult>('/analyze', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  } catch (err) {
    throw parseError(err);
  }
}

export async function analyzeDemo(): Promise<AnalysisResult> {
  try {
    const { data } = await client.post<AnalysisResult>('/analyze/demo');
    return data;
  } catch (err) {
    throw parseError(err);
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const { data } = await client.get('/health');
    return data.status === 'ok';
  } catch {
    return false;
  }
}

function parseError(err: unknown): Error {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (detail) return new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    if (err.code === 'ECONNABORTED') return new Error('Request timed out. The server may be starting up — please retry.');
    if (!err.response) return new Error('Could not reach the server. Check your internet connection.');
  }
  return err instanceof Error ? err : new Error(String(err));
}
