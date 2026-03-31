# ECG Interpreter — Mobile Beta Deploy Plan

> **Goal:** Ship the ECG Interpreter as a native iOS + Android app to beta testers (doctors and health professionals) for feedback and rating collection.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            React Native (Expo) App               │
│  iOS (TestFlight) + Android (Play Internal)      │
└────────────────────┬────────────────────────────┘
                     │ HTTPS  (multipart upload)
                     ▼
┌─────────────────────────────────────────────────┐
│          FastAPI Backend  (api/main.py)          │
│   Hosted on Railway / Render / your own server  │
│                                                  │
│   POST /analyze        ← ECG image upload        │
│   POST /analyze/demo   ← built-in sample ECG     │
│   GET  /health         ← liveness check          │
└────────────────────┬────────────────────────────┘
                     │ internal Python imports
                     ▼
┌─────────────────────────────────────────────────┐
│  Existing ML Pipeline                           │
│  digitizer/pipeline.py → app/inference.py       │
│  checkpoints/best_model.pt (EchoNext)           │
│  checkpoints/ptbxl_best_model.pt (PTB-XL)       │
└─────────────────────────────────────────────────┘
```

---

## Phase 1 — Backend API Deployment

### 1.1 Install API dependencies

```bash
# From the project root
pip install -r requirements.txt          # existing ML deps
pip install -r api/requirements-api.txt  # FastAPI + uvicorn
```

### 1.2 Test locally

```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs to verify the Swagger UI
```

### 1.3 Choose a hosting platform

| Platform | Free tier | GPU support | Notes |
|----------|-----------|-------------|-------|
| **Railway** | 500 hr/mo | No | Easiest — 1-click GitHub deploy |
| **Render** | 750 hr/mo | No | Auto-sleep after 15 min inactivity |
| **Fly.io** | 3 shared VMs | No | Good cold-start times |
| **AWS EC2 / GCP** | Paid | Yes (optional) | Best for production scale |

**Recommended for beta: Railway**

### 1.4 Deploy to Railway

1. Push the repository to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select the repository.
4. Set the **Start Command**:
   ```
   uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables if needed (none required for defaults).
6. Railway assigns a public URL like `https://ecg-interpreter-production.up.railway.app`.
7. Test: `curl https://<your-url>/health`

### 1.5 Copy the API URL

You will need this URL in the next phase.

---

## Phase 2 — Mobile App Configuration

### 2.1 Prerequisites

```bash
node --version    # 18+
npm --version     # 9+
```

Install Expo CLI and EAS CLI:
```bash
npm install -g expo-cli eas-cli
```

### 2.2 Install dependencies

```bash
cd mobile
npm install
```

### 2.3 Set the API URL

Edit `mobile/constants/Config.ts`:
```typescript
API_BASE_URL: 'https://<your-railway-url>',
```

**Or** use an environment variable (preferred for CI/CD):
```bash
# Create mobile/.env
EXPO_PUBLIC_API_URL=https://<your-railway-url>
```

### 2.4 (Optional) Set up the feedback form

1. Create a Google Form with fields: Rating (1-5), Correct? (Y/N/Partial), Comments.
2. Copy the form's prefill URL into `mobile/constants/Config.ts` → `FEEDBACK_FORM_URL`.

---

## Phase 3 — Expo Account & EAS Setup

### 3.1 Create an Expo account

Sign up at [expo.dev](https://expo.dev) (free).

### 3.2 Log in

```bash
eas login
```

### 3.3 Configure EAS for the project

```bash
cd mobile
eas build:configure
```

This links the local project to your Expo account and sets the project ID in `app.json`.

### 3.4 Verify app identifiers

Edit `mobile/app.json` and confirm:
- `ios.bundleIdentifier`: e.g. `com.yourname.ecginterpreter`
- `android.package`: e.g. `com.yourname.ecginterpreter`

These must be globally unique in the App Store / Play Store.

---

## Phase 4 — iOS Beta (TestFlight)

### 4.1 Prerequisites

- Apple Developer account ($99/year) at [developer.apple.com](https://developer.apple.com).
- An app record created in **App Store Connect** → **Apps** → **+**.

### 4.2 Build for iOS

```bash
cd mobile
eas build --platform ios --profile preview
```

EAS builds in the cloud — no Mac required. The build takes ~10–15 minutes.

### 4.3 Submit to TestFlight

When the build finishes, submit it:
```bash
eas submit --platform ios --latest
```

Or in App Store Connect: **TestFlight** tab → upload the `.ipa` file manually.

### 4.4 Invite beta testers

In App Store Connect → **TestFlight** → **External Testing**:
1. Create a group (e.g. "Cardiologists Beta").
2. Click **Add Testers** → enter email addresses of doctors.
3. Each tester receives an email with a TestFlight invitation link.
4. They install TestFlight from the App Store, then tap the invitation link.

**Apple's review for TestFlight external testers takes 1–3 business days.**
Internal testers (your team, up to 100 accounts) are available immediately.

---

## Phase 5 — Android Beta (Google Play Internal Testing)

### 5.1 Prerequisites

- Google Play Developer account ($25 one-time) at [play.google.com/console](https://play.google.com/console).
- Create an app in the Play Console → **All apps** → **Create app**.

### 5.2 Build for Android

```bash
cd mobile
eas build --platform android --profile preview
```

This produces an `.apk` (preview profile) suitable for sideloading, or an `.aab` (production profile) for Play Store.

### 5.3 Upload to Play Console

1. Play Console → your app → **Testing** → **Internal testing** → **Create new release**.
2. Upload the `.aab` / `.apk` file.
3. Save and roll out the release (100%).

### 5.4 Invite beta testers

1. **Internal testing** → **Testers** tab → **Add email list**.
2. Add up to 100 email addresses.
3. Copy the **opt-in URL** and share it with testers.
4. Testers follow the link, join the program, and then install via Play Store.

**Internal testing is available instantly** (no review required).

---

## Phase 6 — Over-the-Air Updates (OTA)

For minor JS/UI changes that don't touch native code, use EAS Update to push fixes instantly without rebuilding or going through store review.

```bash
# After making changes to mobile/ JS files:
eas update --branch production --message "Fix waveform display"
```

Users receive the update silently next time they open the app.

**OTA updates cannot change native modules or app permissions** — those require a full rebuild.

---

## Phase 7 — Beta Feedback Collection

### 7.1 In-app feedback (FeedbackModal)

The app includes a built-in `FeedbackModal` component that captures:
- Overall rating (1–5 stars)
- Clinical correctness (Yes / No / Partial)
- Free-text comments

Currently logs to console. To collect responses, **choose one**:

**Option A — Google Sheets (simplest)**
1. Create a Google Form, enable response collection to a Sheet.
2. Use the Form's POST submission URL in `FeedbackModal.handleSubmit()`:
   ```typescript
   await fetch('https://docs.google.com/forms/d/e/<FORM_ID>/formResponse', {
     method: 'POST',
     body: new URLSearchParams({
       'entry.XXXXXXXX': String(rating),
       'entry.YYYYYYYY': String(correct),
       'entry.ZZZZZZZZ': comments,
     }),
   });
   ```

**Option B — Backend endpoint**
Add `POST /feedback` to `api/main.py` and store to a PostgreSQL database.

**Option C — Third-party (Typeform / Airtable)**
Generate a pre-filled URL and open it with `Linking.openURL()`.

### 7.2 Recruiting beta testers

Share these links with medical institutions / colleagues:
- **iOS**: TestFlight invitation link (from App Store Connect)
- **Android**: Play Internal Testing opt-in URL (from Play Console)

Suggested outreach channels:
- Cardiology department email lists
- LinkedIn (connect with cardiologists, GP networks)
- Medical Twitter/X communities

---

## Phase 8 — Monitoring & Iteration

### 8.1 Error tracking (recommended)

Integrate [Sentry](https://sentry.io) for crash reporting:
```bash
npx expo install @sentry/react-native
```

### 8.2 Analytics (optional)

Use [Expo Analytics](https://docs.expo.dev) or [PostHog](https://posthog.com) to track:
- Number of ECGs analysed
- Risk level distribution
- Average rating submitted

### 8.3 Backend health

Monitor the API with:
```bash
curl https://<your-api-url>/health
```

Set up an uptime monitor (e.g. UptimeRobot — free tier) to alert you if the server goes down.

---

## Quick Reference Checklist

### One-time setup
- [ ] Deploy backend to Railway (or equivalent)
- [ ] Note the public API URL
- [ ] Update `mobile/constants/Config.ts` with the API URL
- [ ] Create Apple Developer account + app record in App Store Connect
- [ ] Create Google Play Developer account + app in Play Console
- [ ] Create Expo account and run `eas build:configure`
- [ ] (Optional) Set up Google Form for feedback

### Each beta release
- [ ] `eas build --platform ios --profile preview` → submit to TestFlight
- [ ] `eas build --platform android --profile preview` → upload to Play Internal Testing
- [ ] Share TestFlight link / Play opt-in URL with testers
- [ ] Monitor feedback responses and server health

### Minor hotfixes (JS only)
- [ ] `eas update --branch production --message "Description"`

---

## File Reference

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI REST API wrapping the Python ML pipeline |
| `api/requirements-api.txt` | FastAPI + uvicorn pip dependencies |
| `mobile/app/index.tsx` | Home screen — upload or demo |
| `mobile/app/analyze.tsx` | Patient info entry + analysis trigger |
| `mobile/app/results.tsx` | Full results display |
| `mobile/components/RiskBadge.tsx` | HIGH/MODERATE/LOW risk badge |
| `mobile/components/ECGWaveform.tsx` | 12-lead SVG waveform viewer |
| `mobile/components/DiagnosisCard.tsx` | Expandable diagnosis card |
| `mobile/components/ConditionBar.tsx` | Condition probability bar |
| `mobile/components/FeedbackModal.tsx` | In-app clinician feedback form |
| `mobile/services/api.ts` | Axios API client |
| `mobile/constants/Config.ts` | API URL + app configuration |
| `mobile/eas.json` | EAS Build + Submit profiles |
