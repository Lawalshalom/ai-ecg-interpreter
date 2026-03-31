import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  SafeAreaView,
  Image,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { Config } from '@/constants/Config';
import { healthCheck } from '@/services/api';
import type { UploadState } from '@/types';

export default function HomeScreen() {
  const router = useRouter();
  const [serverReady, setServerReady] = useState<boolean | null>(null);

  useEffect(() => {
    healthCheck().then(ok => setServerReady(ok));
  }, []);

  async function pickFromCamera() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Camera access is required to photograph ECGs.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.95,
      allowsEditing: false,
    });
    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      const upload: UploadState = {
        uri: asset.uri,
        name: `ecg_${Date.now()}.jpg`,
        mimeType: 'image/jpeg',
      };
      navigateToAnalyze(upload);
    }
  }

  async function pickFromLibrary() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Photo library access is required to upload ECGs.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
      allowsEditing: false,
    });
    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      const ext = asset.uri.split('.').pop()?.toLowerCase() ?? 'jpg';
      const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
      const upload: UploadState = {
        uri: asset.uri,
        name: asset.fileName ?? `ecg_${Date.now()}.${ext}`,
        mimeType,
      };
      navigateToAnalyze(upload);
    }
  }

  async function pickDocument() {
    const result = await DocumentPicker.getDocumentAsync({
      type: ['image/*', 'application/pdf'],
      copyToCacheDirectory: true,
    });
    if (result.assets && result.assets[0]) {
      const asset = result.assets[0];
      const upload: UploadState = {
        uri: asset.uri,
        name: asset.name,
        mimeType: asset.mimeType ?? 'image/jpeg',
      };
      navigateToAnalyze(upload);
    }
  }

  function navigateToAnalyze(upload: UploadState) {
    router.push({
      pathname: '/analyze',
      params: {
        uri: upload.uri,
        name: upload.name,
        mimeType: upload.mimeType,
      },
    });
  }

  function goToDemo() {
    router.push({
      pathname: '/analyze',
      params: { demo: '1' },
    });
  }

  return (
    <SafeAreaView style={styles.root}>
      <LinearGradient colors={[Colors.primary, Colors.primaryLight]} style={styles.heroGradient}>
        <View style={styles.hero}>
          <Ionicons name="pulse" size={48} color="rgba(255,255,255,0.9)" style={styles.heroIcon} />
          <Text style={styles.heroTitle}>ECG Interpreter</Text>
          <Text style={styles.heroSubtitle}>
            AI-powered structural heart disease analysis
          </Text>
          <View style={styles.versionPill}>
            <Text style={styles.versionText}>{Config.APP_VERSION}</Text>
          </View>
        </View>
      </LinearGradient>

      <View style={styles.content}>
        {/* Server status */}
        <View style={[styles.statusRow, serverReady === false && styles.statusError]}>
          <Ionicons
            name={serverReady === true ? 'checkmark-circle' : serverReady === false ? 'alert-circle' : 'ellipse'}
            size={14}
            color={serverReady === true ? Colors.success : serverReady === false ? Colors.danger : Colors.textDisabled}
          />
          <Text style={[styles.statusText, serverReady === false && { color: Colors.danger }]}>
            {serverReady === true
              ? 'AI models ready'
              : serverReady === false
              ? 'Server unreachable — check API URL in Config.ts'
              : 'Connecting…'}
          </Text>
        </View>

        <Text style={styles.sectionLabel}>Upload an ECG</Text>

        <View style={styles.buttonGroup}>
          <TouchableOpacity style={styles.primaryBtn} onPress={pickFromCamera}>
            <Ionicons name="camera" size={22} color={Colors.white} />
            <Text style={styles.primaryBtnText}>Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondaryBtn} onPress={pickFromLibrary}>
            <Ionicons name="images-outline" size={22} color={Colors.primary} />
            <Text style={styles.secondaryBtnText}>Photo Library</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondaryBtn} onPress={pickDocument}>
            <Ionicons name="document-outline" size={22} color={Colors.primary} />
            <Text style={styles.secondaryBtnText}>PDF / File</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.dividerRow}>
          <View style={styles.divider} />
          <Text style={styles.orText}>or</Text>
          <View style={styles.divider} />
        </View>

        <TouchableOpacity style={styles.demoBtn} onPress={goToDemo}>
          <Ionicons name="flask-outline" size={20} color={Colors.primary} />
          <Text style={styles.demoBtnText}>Try with a Demo ECG</Text>
        </TouchableOpacity>

        {/* Instructions */}
        <View style={styles.tips}>
          <Text style={styles.tipsTitle}>Tips for best results</Text>
          {[
            'Use a clear, in-focus landscape photo of the ECG printout',
            'Ensure all 12 leads are visible and not cropped',
            'Avoid glare — photograph under even lighting',
            'Accepted formats: JPEG, PNG, PDF',
          ].map((tip, i) => (
            <View key={i} style={styles.tipRow}>
              <Ionicons name="checkmark" size={14} color={Colors.primary} style={styles.tipIcon} />
              <Text style={styles.tipText}>{tip}</Text>
            </View>
          ))}
        </View>

        <Text style={styles.disclaimer}>{Config.MEDICAL_DISCLAIMER}</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  heroGradient: { paddingBottom: 28 },
  hero: { alignItems: 'center', paddingTop: 48, paddingHorizontal: 24 },
  heroIcon: { marginBottom: 12 },
  heroTitle: {
    fontSize: 30,
    fontWeight: '800',
    color: Colors.white,
    letterSpacing: -0.5,
  },
  heroSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.75)',
    marginTop: 6,
    textAlign: 'center',
  },
  versionPill: {
    marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  versionText: { color: 'rgba(255,255,255,0.9)', fontSize: 12, fontWeight: '600' },
  content: { flex: 1, padding: 20, paddingTop: 16 },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 7,
    marginBottom: 16,
    gap: 6,
  },
  statusError: { backgroundColor: Colors.dangerBg },
  statusText: { fontSize: 12, color: Colors.textSecondary },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: Colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 12,
  },
  buttonGroup: { gap: 10 },
  primaryBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    elevation: 2,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
  },
  primaryBtnText: { color: Colors.white, fontSize: 15, fontWeight: '700' },
  secondaryBtn: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    borderWidth: 1.5,
    borderColor: Colors.border,
  },
  secondaryBtnText: { color: Colors.primary, fontSize: 15, fontWeight: '600' },
  dividerRow: { flexDirection: 'row', alignItems: 'center', marginVertical: 16 },
  divider: { flex: 1, height: 1, backgroundColor: Colors.border },
  orText: { marginHorizontal: 12, fontSize: 13, color: Colors.textDisabled },
  demoBtn: {
    borderRadius: 12,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1.5,
    borderColor: Colors.primaryLight,
    borderStyle: 'dashed',
  },
  demoBtnText: { color: Colors.primary, fontSize: 14, fontWeight: '600' },
  tips: {
    marginTop: 20,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  tipsTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 8,
  },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 5 },
  tipIcon: { marginRight: 6, marginTop: 1 },
  tipText: { flex: 1, fontSize: 12, color: Colors.textSecondary, lineHeight: 18 },
  disclaimer: {
    marginTop: 16,
    fontSize: 11,
    color: Colors.textDisabled,
    textAlign: 'center',
    lineHeight: 16,
  },
});
