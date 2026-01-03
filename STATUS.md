# Status

## Current Phase: Foundation Setup

### ✅ Completed

- [x] Snapdragon 8 RDK X5 hardware arrived (2 weeks early)
- [x] Project repo created in Gitea
- [x] Code skeleton written (3 scripts + docs)
- [x] Architecture designed:
  - Phase 1: Training classifier before Sebring (target: ~2 weeks)
  - Phase 2: Hotel inference + XMP integration (at Sebring)
  - Phase 3: Real-time Z9 attachment (post-Sebring, optional nice-to-have)

### 🚧 In Progress

- [ ] Snapdragon board physical setup
- [ ] Development environment (Python, PyTorch)
- [ ] Lightroom frame export (interesting + boring samples)

### ⏭️ Next Steps (This Week)

1. **Hardware**: Unbox, boot, verify camera module
2. **Environment**: Python 3.11+, PyTorch with appropriate device support (CPU/GPU/MPS)
3. **Data**: Export 275 Lime Rock interesting frames + ~2000-3000 boring frames to organized directories
4. **Execution**: Run `prepare_training_data.py` to create train/test splits

### Timeline

| Phase | Task | Target | Status |
|-------|------|--------|--------|
| 1 | Data prep | This week | Pending |
| 1 | Training | 1 week after data | Not started |
| 1 | Accuracy eval | 2 weeks before Sebring | Not started |
| 2 | Hotel workflow | At Sebring | Not started |
| 2 | XMP integration | At Sebring | Not started |
| 3 | Real-time Z9 | Post-Sebring | Not started |

### Known Issues

- None yet (hardware just arrived)

### Questions/Decisions Needed

- [ ] ImageMagick installed on M4 Max for NEF preview extraction? (If not, will need `brew install imagemagick`)
- [ ] Preference on PyTorch device (MPS for Apple Silicon, or CPU)?
- [ ] Exact Sebring date (needed for scheduling)?

## Notes

- Class imbalance is expected and handled in loss function
- 275 interesting frames from Lime Rock + ~3000 boring gives solid training signal
- Sebring shooting ~10,000 frames from Z9 + ~10,000 from Vic's Z6III
- Hotel workflow (batch classification + XMP) is the MVP; real-time Z9 is future-nice-to-have
