    # RVC Training Loss Tracking Updates

## Summary

Updated the RVC training pipeline to use **correct loss metrics** for training decisions, following best practices for voice conversion model training.

## Key Changes

### ✅ What Changed

#### 1. **Primary Metric: MEL Loss** (Previously: Generator Loss)
- **Before**: Used `loss_gen` (adversarial generator loss) for all decisions
- **After**: Uses `loss_mel` (log-mel reconstruction loss) as PRIMARY metric
- **Why**: Mel loss directly tracks intelligibility and timbre reconstruction quality, while gen loss oscillates and doesn't reliably indicate model quality

#### 2. **Secondary Metric: Feature-Matching Loss**
- Added `loss_fm` (feature-matching) as SECONDARY metric
- Tracks naturalness and artifact reduction
- Correlates with perceptual quality

#### 3. **Composite Score Implementation**
- New formula: `composite = mel + 0.3 × fm`
- Combines intelligibility (mel) with naturalness (fm)
- Weights are adjustable (0.2-0.5 typical for FM weight)
- Used for quality assessment alongside mel

#### 4. **Rolling Window Plateau Detection**
- Tracks mel loss over 8-10 epoch window
- Detects plateau when improvement < 1% over window
- Replaced old gen-based slope detection

#### 5. **Overfitting Detection**
- Monitors if mel loss increases for 5+ consecutive epochs
- Early warning sign that model is memorizing training data
- Triggers early stopping before quality degrades

#### 6. **Updated Early Stopping Logic**
```python
Stop conditions:
1. Mel hasn't improved by ≥1% over last 8-10 epochs (plateau)
2. Mel trending upward for ≥5 consecutive epochs (overfitting)
3. Near-zero mel loss (perfect reconstruction, rare)
```

#### 7. **Checkpoint Saving Based on Mel**
- Best models saved based on mel loss improvements
- Requires ≥1% improvement (configurable)
- Tracks best_mel and best_composite (not best_gen)

### ❌ What's Ignored Now

These losses are **still tracked and logged** but **NOT used for training decisions**:

- `loss_gen` (generator loss) - Adversarial signal, oscillates
- `loss_disc` (discriminator loss) - Adversarial signal, oscillates
- `loss_kl` (KL divergence) - Health monitor only, not quality metric

## Technical Details

### LossTracker Class Updates

#### New Parameters
```python
LossTracker(
    ema_alpha=0.05,                           # EMA smoothing factor
    min_delta=1e-4,                           # Minimum improvement threshold
    min_save_interval=5,                      # Min epochs between saves
    significant_improvement_threshold=0.01,   # 1% improvement required
    max_best_saves=3,                         # Keep best N checkpoints
    total_epochs=100,                         # Total training epochs
    warmup_ratio=0.25,                        # No saves during first 25%
    plateau_patience_epochs=10,               # Rolling window size
    composite_weight_fm=0.3                   # FM weight in composite
)
```

#### New Methods
- `compute_composite()` - Calculate mel + 0.3 × fm
- `on_epoch_end(epoch)` - Update rolling window tracking
- Enhanced `should_early_stop()` - Mel-based plateau/overfit detection
- Enhanced `should_save_best()` - Mel-based checkpoint saving
- Enhanced `status_str()` - Highlight mel/fm over gen/disc

#### New Tracking Variables
```python
self.best_mel                    # Best mel loss seen
self.best_composite              # Best composite score
self.mel_history                 # Rolling window (10 epochs)
self.composite_history           # Rolling window (10 epochs)
self.mel_uptrend_epochs          # Count of consecutive increases
self.composite_weight_fm = 0.3   # Configurable FM weight
```

### Training Loop Changes

#### Log Output Format
**Before:**
```
EMA(gen=4.19 disc=3.51 mel=23.60 kl=0.45 fm=13.43)
```

**After:**
```
EMA(mel=22.38 [best=21.50], fm=13.43, composite=26.41 [best=25.93]) | 
gen=3.51 disc=3.40 kl=0.45 | improvement_over_10ep=5.12%, 
uptrend_epochs=0, epochs_since_best_save=3
```

#### Early Stopping Messages
**Before:**
```
[Tracker] Early stopping: sustained upslope or plateau detected.
```

**After:**
```
[Tracker] Early stopping: mel loss plateau (<1% improvement over 10 epochs) or overfitting detected.
```

## Configuration Recommendations

### Default Settings (Works for Most Cases)
- `significant_improvement_threshold=0.01` (1%)
- `plateau_patience_epochs=10`
- `composite_weight_fm=0.3`
- `max_uptrend_patience=5`

### For High-Quality Models (More Patience)
- `significant_improvement_threshold=0.005` (0.5%)
- `plateau_patience_epochs=15`
- `composite_weight_fm=0.4`
- `max_uptrend_patience=8`

### For Fast Iteration (Less Patience)
- `significant_improvement_threshold=0.02` (2%)
- `plateau_patience_epochs=8`
- `composite_weight_fm=0.2`
- `max_uptrend_patience=3`

## Reading the Logs

### What to Watch

1. **Primary**: `mel` value and `[best=X]`
   - Lower is better
   - Should steadily decrease
   - If rising, potential overfitting

2. **Secondary**: `fm` (feature-matching)
   - Lower is better
   - Complements mel for quality

3. **Composite**: `composite` and `[best=X]`
   - Combined quality metric
   - mel + 0.3 × fm

4. **Improvement**: `improvement_over_10ep`
   - Should be > 1% to continue training
   - < 1% triggers plateau detection

5. **Uptrend**: `uptrend_epochs`
   - Counts consecutive epochs where mel increases
   - ≥ 5 triggers overfitting early stop

### What to Ignore for Decisions

- `gen` (generator loss) - Will oscillate, that's normal
- `disc` (discriminator loss) - Will oscillate, that's normal
- `kl` (KL divergence) - Health monitor, not quality

### Example Interpretation

```
EMA(mel=22.38 [best=21.50], fm=13.43, composite=26.41 [best=25.93]) | 
gen=3.51 disc=3.40 kl=0.45 | improvement_over_10ep=5.12%, 
uptrend_epochs=0, epochs_since_best_save=3
```

**Analysis:**
- ✅ Mel loss: 22.38 (current) vs 21.50 (best) - Not at best yet but close
- ✅ Composite: 26.41 (current) vs 25.93 (best) - Similarly tracking
- ✅ Improvement: 5.12% over 10 epochs - GOOD, well above 1% threshold
- ✅ Uptrend: 0 epochs - No overfitting detected
- ✅ Gen/Disc: Oscillating as expected (ignore for decisions)
- **Verdict**: Training is progressing well, continue

## Files Modified

1. **`modules/rvc/infer/modules/train/train.py`**
   - Lines 57-307: Complete LossTracker class rewrite
   - Lines 690-702: Updated initialization parameters
   - Lines 714-719: Updated early stopping logic and messages
   - Lines 744-747: Updated near-zero loss message

## Validation Checklist

- [x] Mel loss used as primary metric
- [x] Feature-matching loss used as secondary metric
- [x] Composite score implemented (mel + 0.3 × fm)
- [x] Rolling window tracking (8-10 epochs)
- [x] 1% improvement threshold implemented
- [x] Overfitting detection (5 epoch uptrend)
- [x] Gen/disc ignored for plateau decisions
- [x] KL treated as health indicator only
- [x] Checkpoint saving based on mel improvements
- [x] Enhanced logging shows mel/fm prominently
- [x] Early stopping messages clarified

## Migration Notes

### No Breaking Changes
- Existing training configs will work
- Old checkpoints remain compatible
- Only decision logic changed, not model architecture

### Behavioral Changes
- Training may stop earlier (better plateau detection)
- Checkpoints saved at different times (based on mel, not gen)
- Logs show mel/fm prominently with composite scores

## Benefits

1. **Better Model Quality**: Decisions based on actual reconstruction quality (mel)
2. **Prevent Overfitting**: Early detection of mel uptrend
3. **Clearer Logging**: Primary metrics (mel/fm) shown first
4. **Smarter Checkpoints**: Save when quality actually improves
5. **Fewer Wasted Epochs**: Better plateau detection stops earlier
6. **Standard Practice**: Aligns with voice conversion research best practices

## References

- Log-mel reconstruction loss: Primary metric for speech/voice quality
- Feature-matching loss: Perceptual quality for GANs (Larsen et al.)
- Composite metrics: Common in voice conversion papers
- Early stopping on validation metrics: Standard ML practice

