/**
 * 12-lead ECG Waveform Viewer
 * Renders a 3-row × 4-column clinical ECG grid using react-native-svg.
 * Input: waveform[12][500] (downsampled from 2500 at 5x)
 */
import React, { useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import Svg, { Path, Line, Rect, Text as SvgText } from 'react-native-svg';
import { Colors } from '@/constants/Colors';

interface Props {
  waveform: number[][];   // 12 × 500
  leadNames: string[];    // 12 names
}

// Standard 12-lead layout: row → [lead index, lead name]
const LAYOUT = [
  [0, 3, 6, 9],   // Row 1: I, aVR, V1, V4
  [1, 4, 7, 10],  // Row 2: II, aVL, V2, V5
  [2, 5, 8, 11],  // Row 3: III, aVF, V3, V6
];

const CELL_W = 200;
const CELL_H = 80;
const LABEL_H = 14;
const PADDING = 4;
const ROWS = 3;
const COLS = 4;
const SVG_W = CELL_W * COLS;
const SVG_H = (CELL_H + LABEL_H) * ROWS;

function buildPath(samples: number[], cellX: number, cellY: number): string {
  if (!samples || samples.length === 0) return '';

  // Normalise to cell height
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const range = max - min || 1;

  const usableH = CELL_H - PADDING * 2;
  const stepX = CELL_W / (samples.length - 1);

  let d = '';
  for (let i = 0; i < samples.length; i++) {
    const x = cellX + i * stepX;
    const y = cellY + LABEL_H + PADDING + usableH * (1 - (samples[i] - min) / range);
    d += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

export default function ECGWaveform({ waveform, leadNames }: Props) {
  const paths = useMemo(() => {
    const result: { d: string; label: string; x: number; y: number }[] = [];
    LAYOUT.forEach((row, rowIdx) => {
      row.forEach((leadIdx, colIdx) => {
        const cellX = colIdx * CELL_W;
        const cellY = rowIdx * (CELL_H + LABEL_H);
        result.push({
          d: buildPath(waveform[leadIdx] ?? [], cellX, cellY),
          label: leadNames[leadIdx] ?? `L${leadIdx + 1}`,
          x: cellX,
          y: cellY,
        });
      });
    });
    return result;
  }, [waveform, leadNames]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>12-Lead ECG</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <Svg width={SVG_W} height={SVG_H} style={styles.svg}>
          {/* Grid background cells */}
          {paths.map((p, i) => (
            <Rect
              key={`bg-${i}`}
              x={p.x}
              y={p.y}
              width={CELL_W}
              height={CELL_H + LABEL_H}
              fill={Colors.ecgBackground}
              stroke={Colors.border}
              strokeWidth={0.5}
            />
          ))}

          {/* Major grid lines (5mm = 0.2s) */}
          {Array.from({ length: ROWS }).map((_, r) =>
            Array.from({ length: COLS }).map((_, c) => {
              const x0 = c * CELL_W;
              const y0 = r * (CELL_H + LABEL_H) + LABEL_H;
              const lines = [];
              for (let gx = 0; gx <= 5; gx++) {
                lines.push(
                  <Line
                    key={`gl-${r}-${c}-${gx}`}
                    x1={x0 + (gx * CELL_W) / 5}
                    y1={y0}
                    x2={x0 + (gx * CELL_W) / 5}
                    y2={y0 + CELL_H}
                    stroke={Colors.ecgGrid}
                    strokeWidth={0.5}
                  />
                );
              }
              return lines;
            })
          )}

          {/* Lead labels */}
          {paths.map((p, i) => (
            <SvgText
              key={`lbl-${i}`}
              x={p.x + 5}
              y={p.y + LABEL_H - 2}
              fontSize={10}
              fontWeight="bold"
              fill={Colors.primary}
            >
              {p.label}
            </SvgText>
          ))}

          {/* ECG traces */}
          {paths.map((p, i) => (
            <Path
              key={`trace-${i}`}
              d={p.d}
              stroke={Colors.ecgTrace}
              strokeWidth={1.2}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </Svg>
      </ScrollView>
      <Text style={styles.hint}>Scroll horizontally · Tap to zoom</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 12,
    marginVertical: 8,
    elevation: 1,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
  },
  title: {
    fontSize: 13,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  svg: {
    borderRadius: 6,
    overflow: 'hidden',
  },
  hint: {
    marginTop: 6,
    fontSize: 11,
    color: Colors.textDisabled,
    textAlign: 'center',
  },
});
