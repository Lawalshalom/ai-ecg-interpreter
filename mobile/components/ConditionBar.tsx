import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '@/constants/Colors';

interface Props {
  label: string;
  probability: number;
  isComposite?: boolean;
}

function barColor(p: number): string {
  if (p >= 0.6) return Colors.danger;
  if (p >= 0.35) return Colors.warning;
  return Colors.success;
}

export default function ConditionBar({ label, probability, isComposite }: Props) {
  const pct = Math.round(probability * 100);
  const color = barColor(probability);

  return (
    <View style={[styles.row, isComposite && styles.compositeRow]}>
      <Text style={[styles.label, isComposite && styles.compositeLabel]} numberOfLines={2}>
        {label}
      </Text>
      <View style={styles.barContainer}>
        <View style={[styles.bar, { width: `${pct}%`, backgroundColor: color }]} />
      </View>
      <Text style={[styles.pct, { color }]}>{pct}%</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
    paddingHorizontal: 12,
  },
  compositeRow: {
    backgroundColor: Colors.surfaceVariant,
    borderRadius: 8,
    marginBottom: 4,
  },
  label: {
    width: 130,
    fontSize: 12,
    color: Colors.textPrimary,
    flexShrink: 0,
  },
  compositeLabel: {
    fontWeight: '700',
  },
  barContainer: {
    flex: 1,
    height: 8,
    backgroundColor: Colors.border,
    borderRadius: 4,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  bar: {
    height: 8,
    borderRadius: 4,
  },
  pct: {
    width: 38,
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'right',
  },
});
