# Demo-day script (3-minute video + live pitch)

## Shot list

| Time | Screen | Say |
|---|---|---|
| 0:00 | A cloud redaction tool, pasting a bank statement | "This is how India redacts documents today — by uploading an Aadhaar to someone else's server." |
| 0:20 | Redacta UI, paste the same doc, hit **Scan** | "Redacta AI never uploads anything. Watch the detections." |
| 0:35 | Findings tab | "Aadhaar validated with the UIDAI Verhoeff checksum. PAN holder-letter. Luhn-checked card. Every hit shows *why*." |
| 1:00 | Redacted output + token hints | "Redacted — and reversibly: tokens replace the real values, originals stay encrypted in a local vault." |
| 1:25 | Terminal: `reveal` with passphrase → original reappears | "Authorised workflows can re-identify. Everyone else sees tokens." |
| 1:50 | Wireshark capture running while using the app | "Packet capture during the whole demo: zero bytes leave this machine." |
| 2:10 | Slide: architecture + AI Hub profiling table | "The NER model compiles through Qualcomm AI Hub to the Hexagon NPU of Snapdragon X-powered HP PCs — this is the private-AI story, shipped." |
| 2:40 | Slide: use cases | "BPOs, banks, hospitals, freelancers — anyone sharing KYC documents daily." |
| 2:55 | Logo + "Redacta AI — privacy that never leaves the device" | Close. |

## Recording checklist

- [ ] Fresh sample doc (`python app.py keygen`) — synthetic data only.
- [ ] Wireshark filter pre-set, window visible during the UI demo.
- [ ] Font size ≥ 16 in terminal and browser.
- [ ] Record at 1080p; test audio levels; keep under 3:00.
- [ ] End card: name, entry ID, "Built for Snapdragon® AI Lab Challenge".
