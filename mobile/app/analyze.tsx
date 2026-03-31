import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { analyzeECG, analyzeDemo } from '@/services/api';
import type { PatientInfo, AnalysisResult } from '@/types';

export default function AnalyzeScreen() {
  const params = useLocalSearchParams<{ uri?: string; name?: string; mimeType?: string; demo?: string }>();
  const router = useRouter();

  const isDemo = params.demo === '1';
  const [age, setAge] = useState<number | undefined>();
  const [sex, setSex] = useState<'Male' | 'Female' | undefined>();
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');

  async function handleAnalyze() {
    setLoading(true);
    try {
      let result: AnalysisResult;

      if (isDemo) {
        setLoadingStep('Loading sample ECG from PTB-XL database…');
        result = await analyzeDemo();
      } else {
        if (!params.uri || !params.name || !params.mimeType) {
          Alert.alert('Error', 'No file selected.');
          return;
        }
        setLoadingStep('Digitizing ECG waveform…');
        const patient: PatientInfo = { age, sex };
        result = await analyzeECG(params.uri, params.name, params.mimeType, patient);
        setLoadingStep('Running AI analysis…');
      }

      // Navigate to results
      router.push({
        pathname: '/results',
        params: { data: JSON.stringify(result) },
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      Alert.alert('Analysis failed', msg);
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  }

  // Auto-start for demo
  useEffect(() => {
    if (isDemo) handleAnalyze();
  }, []);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingTitle}>Analysing ECG…</Text>
        <Text style={styles.loadingStep}>{loadingStep}</Text>
        <Text style={styles.loadingHint}>This usually takes 5 – 15 seconds</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView style={styles.root} contentContainerStyle={styles.content}>
        {/* Image preview */}
        {!isDemo && params.uri && (
          <View style={styles.previewCard}>
            <Image source={{ uri: params.uri }} style={styles.preview} resizeMode="contain" />
            <Text style={styles.previewName} numberOfLines={1}>
              {params.name}
            </Text>
          </View>
        )}

        {isDemo && (
          <View style={styles.demoCard}>
            <Ionicons name="flask" size={28} color={Colors.primary} />
            <Text style={styles.demoText}>Demo ECG from PTB-XL database</Text>
          </View>
        )}

        {/* Patient info */}
        <Text style={styles.sectionLabel}>Patient Demographics (optional)</Text>
        <Text style={styles.sectionHint}>
          Adding age and sex improves feature extraction accuracy.
        </Text>

        {/* Age */}
        <Text style={styles.fieldLabel}>Age</Text>
        <View style={styles.ageRow}>
          {[undefined, 25, 35, 45, 55, 65, 75].map(a => (
            <TouchableOpacity
              key={String(a)}
              style={[styles.ageChip, age === a && styles.ageChipActive]}
              onPress={() => setAge(a)}
            >
              <Text style={[styles.ageChipText, age === a && styles.ageChipTextActive]}>
                {a === undefined ? '—' : a}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Sex */}
        <Text style={styles.fieldLabel}>Sex</Text>
        <View style={styles.sexRow}>
          {(['Male', 'Female', undefined] as const).map(s => (
            <TouchableOpacity
              key={String(s)}
              style={[styles.sexChip, sex === s && styles.sexChipActive]}
              onPress={() => setSex(s)}
            >
              {s === 'Male' && <Ionicons name="male" size={16} color={sex === s ? Colors.white : Colors.primary} />}
              {s === 'Female' && <Ionicons name="female" size={16} color={sex === s ? Colors.white : Colors.primary} />}
              <Text style={[styles.sexChipText, sex === s && styles.sexChipTextActive]}>
                {s ?? 'Unknown'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimerBox}>
          <Ionicons name="information-circle-outline" size={16} color={Colors.textSecondary} />
          <Text style={styles.disclaimerText}>
            For research use only. Not a certified medical device.
            Always confirm findings with a qualified cardiologist.
          </Text>
        </View>

        {/* Analyse button */}
        <TouchableOpacity style={styles.analyzeBtn} onPress={handleAnalyze}>
          <Ionicons name="analytics" size={22} color={Colors.white} />
          <Text style={styles.analyzeBtnText}>Analyse ECG</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  content: { padding: 20, paddingBottom: 40 },
  previewCard: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  preview: { width: '100%', height: 200, backgroundColor: Colors.surfaceVariant },
  previewName: {
    padding: 10,
    fontSize: 12,
    color: Colors.textSecondary,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  demoCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#e8eaf6',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  demoText: { fontSize: 14, color: Colors.primary, fontWeight: '600' },
  sectionLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  sectionHint: {
    fontSize: 13,
    color: Colors.textSecondary,
    marginBottom: 16,
    lineHeight: 18,
  },
  fieldLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 8,
    marginTop: 4,
  },
  ageRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 16 },
  ageChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  ageChipActive: { borderColor: Colors.primary, backgroundColor: '#e8eaf6' },
  ageChipText: { fontSize: 13, color: Colors.textSecondary },
  ageChipTextActive: { color: Colors.primary, fontWeight: '700' },
  sexRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  sexChip: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  sexChipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  sexChipText: { fontSize: 14, color: Colors.primary, fontWeight: '600' },
  sexChipTextActive: { color: Colors.white },
  disclaimerBox: {
    flexDirection: 'row',
    gap: 8,
    backgroundColor: Colors.surfaceVariant,
    borderRadius: 10,
    padding: 12,
    marginBottom: 24,
  },
  disclaimerText: { flex: 1, fontSize: 12, color: Colors.textSecondary, lineHeight: 17 },
  analyzeBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 14,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    elevation: 3,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  analyzeBtnText: { color: Colors.white, fontSize: 16, fontWeight: '700' },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.background,
    padding: 40,
  },
  loadingTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginTop: 20,
  },
  loadingStep: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginTop: 10,
    textAlign: 'center',
  },
  loadingHint: {
    fontSize: 12,
    color: Colors.textDisabled,
    marginTop: 8,
  },
});
