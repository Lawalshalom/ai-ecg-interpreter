import React, { useState } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';

interface Props {
  visible: boolean;
  onClose: () => void;
  riskLevel: string;
  topDiagnosis?: string;
}

export default function FeedbackModal({ visible, onClose, riskLevel, topDiagnosis }: Props) {
  const [rating, setRating] = useState(0);
  const [correct, setCorrect] = useState<boolean | null>(null);
  const [comments, setComments] = useState('');
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit() {
    if (rating === 0) {
      Alert.alert('Rating required', 'Please give an overall rating before submitting.');
      return;
    }
    // In production: POST to your feedback endpoint or embed Google Form logic
    console.log({ rating, correct, comments, riskLevel, topDiagnosis });
    setSubmitted(true);
  }

  function handleClose() {
    setRating(0);
    setCorrect(null);
    setComments('');
    setSubmitted(false);
    onClose();
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={handleClose}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Clinician Feedback</Text>
          <TouchableOpacity onPress={handleClose} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
            <Ionicons name="close" size={24} color={Colors.textSecondary} />
          </TouchableOpacity>
        </View>

        {submitted ? (
          <View style={styles.thankYou}>
            <Ionicons name="checkmark-circle" size={64} color={Colors.success} />
            <Text style={styles.thankYouTitle}>Thank you!</Text>
            <Text style={styles.thankYouText}>
              Your feedback helps us improve the model for all clinicians.
            </Text>
            <TouchableOpacity style={styles.doneBtn} onPress={handleClose}>
              <Text style={styles.doneBtnText}>Done</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <ScrollView contentContainerStyle={styles.form} showsVerticalScrollIndicator={false}>
            <Text style={styles.disclaimer}>
              Your responses are anonymised and used solely to improve the AI model.
            </Text>

            <Text style={styles.fieldLabel}>Overall rating</Text>
            <View style={styles.stars}>
              {[1, 2, 3, 4, 5].map(s => (
                <TouchableOpacity key={s} onPress={() => setRating(s)} style={styles.star}>
                  <Ionicons
                    name={s <= rating ? 'star' : 'star-outline'}
                    size={34}
                    color={s <= rating ? '#f59e0b' : Colors.border}
                  />
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>Was the AI interpretation clinically correct?</Text>
            <View style={styles.toggle}>
              {[
                { label: 'Yes', value: true },
                { label: 'No', value: false },
                { label: 'Partially', value: null },
              ].map(opt => (
                <TouchableOpacity
                  key={String(opt.label)}
                  style={[
                    styles.toggleBtn,
                    correct === opt.value && styles.toggleBtnActive,
                  ]}
                  onPress={() => setCorrect(opt.value)}
                >
                  <Text
                    style={[
                      styles.toggleText,
                      correct === opt.value && styles.toggleTextActive,
                    ]}
                  >
                    {opt.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>Comments (optional)</Text>
            <TextInput
              style={styles.input}
              multiline
              numberOfLines={4}
              placeholder="Describe findings, corrections, or suggestions…"
              placeholderTextColor={Colors.textDisabled}
              value={comments}
              onChangeText={setComments}
              textAlignVertical="top"
            />

            <View style={styles.caseInfo}>
              <Text style={styles.caseInfoText}>Risk level shown: {riskLevel}</Text>
              {topDiagnosis && (
                <Text style={styles.caseInfoText}>Top diagnosis: {topDiagnosis}</Text>
              )}
            </View>

            <TouchableOpacity style={styles.submitBtn} onPress={handleSubmit}>
              <Text style={styles.submitText}>Submit Feedback</Text>
            </TouchableOpacity>
          </ScrollView>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  title: { fontSize: 18, fontWeight: '700', color: Colors.textPrimary },
  form: { padding: 20 },
  disclaimer: {
    fontSize: 12,
    color: Colors.textSecondary,
    backgroundColor: Colors.surfaceVariant,
    borderRadius: 8,
    padding: 10,
    marginBottom: 20,
    lineHeight: 17,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 10,
    marginTop: 6,
  },
  stars: { flexDirection: 'row', marginBottom: 20 },
  star: { marginRight: 8 },
  toggle: { flexDirection: 'row', marginBottom: 20, gap: 10 },
  toggleBtn: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1.5,
    borderColor: Colors.border,
    alignItems: 'center',
  },
  toggleBtnActive: { borderColor: Colors.primary, backgroundColor: '#e8eaf6' },
  toggleText: { fontSize: 13, color: Colors.textSecondary, fontWeight: '500' },
  toggleTextActive: { color: Colors.primary, fontWeight: '700' },
  input: {
    backgroundColor: Colors.surface,
    borderWidth: 1.5,
    borderColor: Colors.border,
    borderRadius: 10,
    padding: 12,
    fontSize: 13,
    color: Colors.textPrimary,
    minHeight: 100,
    marginBottom: 16,
  },
  caseInfo: {
    backgroundColor: Colors.surfaceVariant,
    borderRadius: 8,
    padding: 10,
    marginBottom: 20,
  },
  caseInfoText: { fontSize: 12, color: Colors.textSecondary, marginBottom: 2 },
  submitBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  submitText: { color: Colors.white, fontSize: 15, fontWeight: '700' },
  thankYou: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40 },
  thankYouTitle: { fontSize: 24, fontWeight: '800', color: Colors.textPrimary, marginTop: 16 },
  thankYouText: {
    fontSize: 15,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginTop: 8,
  },
  doneBtn: {
    marginTop: 32,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingHorizontal: 48,
    paddingVertical: 14,
  },
  doneBtnText: { color: Colors.white, fontSize: 15, fontWeight: '700' },
});
