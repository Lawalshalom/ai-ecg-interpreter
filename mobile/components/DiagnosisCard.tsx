import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { Diagnosis } from '@/types';
import { Colors, RiskColors } from '@/constants/Colors';

interface Props {
  diagnosis: Diagnosis;
}

const LEVEL_ICON: Record<string, keyof typeof Ionicons.glyphMap> = {
  HIGH: 'alert-circle',
  MODERATE: 'warning',
  POSSIBLE: 'information-circle',
};

export default function DiagnosisCard({ diagnosis }: Props) {
  const [expanded, setExpanded] = useState(false);
  const c = RiskColors[diagnosis.level as keyof typeof RiskColors] ?? RiskColors.LOW;

  return (
    <TouchableOpacity
      style={[styles.card, { borderLeftColor: c.badge }]}
      onPress={() => setExpanded(v => !v)}
      activeOpacity={0.8}
    >
      <View style={styles.header}>
        <Ionicons name={LEVEL_ICON[diagnosis.level] ?? 'ellipse'} size={20} color={c.badge} style={styles.icon} />
        <View style={styles.headerText}>
          <Text style={styles.name}>{diagnosis.name}</Text>
          <Text style={[styles.level, { color: c.text }]}>
            {diagnosis.level} CONFIDENCE · {Math.round(diagnosis.confidence * 100)}%
          </Text>
        </View>
        <Ionicons
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={18}
          color={Colors.textSecondary}
        />
      </View>

      {expanded && (
        <View style={styles.body}>
          <Text style={styles.description}>{diagnosis.description}</Text>
          {diagnosis.findings.length > 0 && (
            <>
              <Text style={styles.findingsLabel}>Supporting findings:</Text>
              {diagnosis.findings.map((f, i) => (
                <View key={i} style={styles.findingRow}>
                  <Text style={styles.bullet}>•</Text>
                  <Text style={styles.finding}>{f}</Text>
                </View>
              ))}
            </>
          )}
          <View style={styles.sourcePill}>
            <Text style={styles.sourceText}>{diagnosis.source}</Text>
          </View>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderLeftWidth: 4,
    marginVertical: 5,
    elevation: 1,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
  },
  icon: { marginRight: 10 },
  headerText: { flex: 1 },
  name: {
    fontSize: 14,
    fontWeight: '700',
    color: Colors.textPrimary,
  },
  level: {
    fontSize: 11,
    fontWeight: '600',
    marginTop: 2,
    letterSpacing: 0.3,
  },
  body: {
    paddingHorizontal: 14,
    paddingBottom: 14,
  },
  description: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
    marginBottom: 8,
  },
  findingsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  findingRow: {
    flexDirection: 'row',
    marginBottom: 2,
  },
  bullet: {
    color: Colors.primary,
    marginRight: 6,
    fontSize: 13,
  },
  finding: {
    flex: 1,
    fontSize: 12,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  sourcePill: {
    marginTop: 8,
    alignSelf: 'flex-start',
    backgroundColor: Colors.surfaceVariant,
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  sourceText: {
    fontSize: 11,
    color: Colors.textSecondary,
  },
});
