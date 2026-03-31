import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import type { RiskLevel } from '@/types';
import { RiskColors, Colors } from '@/constants/Colors';

interface Props {
  level: RiskLevel;
  probability: number;
}

const LABEL: Record<RiskLevel, string> = {
  HIGH: 'HIGH RISK',
  MODERATE: 'MODERATE RISK',
  LOW: 'LOW RISK',
};

const SUBTITLE: Record<RiskLevel, string> = {
  HIGH: 'Structural heart disease likely',
  MODERATE: 'Possible structural involvement',
  LOW: 'No significant structural findings',
};

export default function RiskBadge({ level, probability }: Props) {
  const c = RiskColors[level];
  const pct = Math.round(probability * 100);

  return (
    <View style={[styles.container, { backgroundColor: c.bg, borderColor: c.border }]}>
      <View style={[styles.pill, { backgroundColor: c.badge }]}>
        <Text style={styles.pillText}>{LABEL[level]}</Text>
      </View>
      <Text style={[styles.pct, { color: c.text }]}>{pct}% SHD probability</Text>
      <Text style={[styles.subtitle, { color: Colors.textSecondary }]}>{SUBTITLE[level]}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    borderWidth: 1.5,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
  },
  pill: {
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 6,
    marginBottom: 10,
  },
  pillText: {
    color: Colors.white,
    fontWeight: '700',
    fontSize: 14,
    letterSpacing: 0.8,
  },
  pct: {
    fontSize: 26,
    fontWeight: '800',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
    textAlign: 'center',
  },
});
