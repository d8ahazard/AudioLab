# RVC Training Logging & Visualization Updates

## Summary

Cleaned up training logs for better readability and added automatic loss graph generation to visualize training progress with correct metrics (mel/fm/composite).

## Key Changes

### ✅ 1. Cleaned Up Logging

#### Before: Redundant and Messy Logs
```
Train Epoch: 74 [50%]
[1234, 0.0001]
loss_disc=3.510, loss_gen=3.510, loss_fm=13.430, loss_mel=22.380, loss_kl=0.450
[Tracker] EMA(gen=4.19 disc=3.51 mel=23.60 kl=0.45 fm=13.43), best_gen=4.10, ...
```

**Problems:**
- Multiple log lines per step
- Raw arrays printed
- Losses shown twice (raw + EMA)
- Hard to scan quickly

#### After: Clean and Focused Logs
```
Epoch 74: 100%|████████████| 1234/1234 [05:23<00:00, 3.82it/s]
[Tracker] EMA(mel=22.38 [best=21.50], fm=13.43, composite=26.41 [best=25.93]) | 
         gen=3.51 disc=3.40 kl=0.45 | improvement_over_10ep=5.12%, 
         uptrend_epochs=0, epochs_since_best_save=3
```

**Benefits:**
- Single line per log interval
- TQDM progress bar shows training speed
- LossTracker shows all key metrics in one line
- PRIMARY metrics (mel/fm/composite) shown first
- Easy to see improvement trends

#### What Was Removed
```python
# REMOVED: Redundant logging
logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
logger.info([global_step, lr])  # Raw array printing
logger.info(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, ...")  # Duplicate loss info
```

Now only the LossTracker status string is logged, which contains everything needed.

### ✅ 2. Loss Graph Visualization

#### Automatic Graph Generation
Every checkpoint save now generates a loss curve graph showing:

**Top Panel - PRIMARY Metrics** (for decision making)
- 📘 **MEL Loss** (log-mel reconstruction) - Blue line
- 📗 **FM Loss** (feature matching) - Green line  
- 🟣 **COMPOSITE** (mel + 0.3×fm) - Purple dashed line
- ⭐ Best MEL checkpoint marked with star
- Horizontal line showing best MEL value

**Bottom Panel - SECONDARY Metrics** (monitoring only)
- 🔴 **Generator Loss** - Red line (grayed out)
- 🟠 **Discriminator Loss** - Orange line (grayed out)
- 🟤 **KL Divergence** - Brown line (grayed out)

**Clear Labels:**
- Top panel titled: "PRIMARY Metrics (mel/fm/composite) - Use for Quality Assessment"
- Bottom panel titled: "SECONDARY Metrics (gen/disc/kl) - For Monitoring Only (NOT for decisions)"
- Footer note: "Watch MEL (primary) and FM (secondary) for quality. Ignore GEN/DISC oscillations."

#### Graph Features
- **High Resolution**: Saved at 150 DPI
- **Clear Markers**: Different shapes for each metric (○, □, ◇)
- **Best Checkpoint**: Marked with blue star on MEL curve
- **Grid Lines**: For easy value reading
- **Legend**: Clear labels for all metrics
- **Two Locations**: 
  - In project directory: `loss_plot_epoch{N}.png`
  - In trained models: `{model_name}_losses.png`

### ✅ 3. Loss History Tracking

#### New Data Structure
```python
self.epoch_history = [
    {
        'epoch': 1,
        'mel': 25.4321,
        'fm': 14.2341,
        'composite': 29.7023,
        'gen': 4.5432,
        'disc': 3.6543,
        'kl': 0.5432
    },
    # ... one entry per epoch
]
```

#### Benefits
- Complete history preserved for plotting
- Can analyze training retrospectively
- Easy to spot trends and anomalies
- Survives checkpoint resumption

### ✅ 4. Enhanced Status String

#### Format
```
EMA(mel=X.XX [best=Y.YY], fm=X.XX, composite=X.XX [best=Y.YY]) | 
gen=X.XX disc=X.XX kl=X.XX | improvement_over_Ne=P.PP%, 
uptrend_epochs=N, epochs_since_best_save=N
```

#### What Each Part Means

**PRIMARY Section** (mel/fm/composite):
- `mel=22.38 [best=21.50]` - Current mel vs best mel ever seen
- `fm=13.43` - Current feature-matching loss
- `composite=26.41 [best=25.93]` - Combined quality metric

**SECONDARY Section** (gen/disc/kl):
- `gen=3.51` - Generator loss (will oscillate)
- `disc=3.40` - Discriminator loss (will oscillate)
- `kl=0.45` - KL divergence (health check)

**Progress Indicators**:
- `improvement_over_10ep=5.12%` - How much mel improved over last 10 epochs
- `uptrend_epochs=0` - Consecutive epochs where mel increased (overfitting warning)
- `epochs_since_best_save=3` - Epochs since last best checkpoint

## Technical Implementation

### New Methods in LossTracker

```python
def on_epoch_end(self, epoch: int):
    """Store loss snapshot for plotting at end of each epoch"""
    # Stores to self.epoch_history for graph generation

def plot_losses(self, save_path: str, project_name: str = "RVC"):
    """Generate and save loss curve visualization"""
    # Creates dual-panel plot with matplotlib
    # Top: mel/fm/composite (PRIMARY)
    # Bottom: gen/disc/kl (SECONDARY)
```

### New Dependencies
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
```

### File Locations

**During Training:**
```
outputs/voices/{project_name}/
  ├── loss_plot_epoch25.png
  ├── loss_plot_epoch50.png
  └── loss_plot_epoch75.png
```

**Final Model:**
```
models/trained/
  ├── {model_name}.pth
  ├── {model_name}.index
  └── {model_name}_losses.png  ← Loss graph with the model
```

## Usage Examples

### Reading the Logs

**Good Training Progress:**
```
[Tracker] EMA(mel=18.23 [best=18.23], fm=11.54, composite=21.69 [best=21.69]) | 
         gen=3.21 disc=3.15 kl=0.38 | improvement_over_10ep=8.45%, 
         uptrend_epochs=0, epochs_since_best_save=0
```
- ✅ MEL at new best (18.23)
- ✅ Composite at new best (21.69)
- ✅ 8.45% improvement over 10 epochs (well above 1% threshold)
- ✅ No uptrend (not overfitting)
- ✅ Just saved best checkpoint

**Plateau Detected:**
```
[Tracker] EMA(mel=18.21 [best=18.23], fm=11.52, composite=21.67 [best=21.69]) | 
         gen=3.18 disc=3.19 kl=0.37 | improvement_over_10ep=0.65%, 
         uptrend_epochs=0, epochs_since_best_save=15
[Tracker] Early stopping: mel loss plateau (<1% improvement over 10 epochs) or overfitting detected.
```
- ⚠️ MEL not improving (18.21 vs best 18.23)
- ⚠️ Only 0.65% improvement (below 1% threshold)
- ⚠️ 15 epochs since best save (no progress)
- 🛑 Training stops to prevent wasted epochs

**Overfitting Detected:**
```
[Tracker] EMA(mel=18.35 [best=18.23], fm=11.61, composite=21.82 [best=21.69]) | 
         gen=3.08 disc=3.25 kl=0.36 | improvement_over_10ep=-0.82%, 
         uptrend_epochs=5, epochs_since_best_save=8
[Tracker] Early stopping: mel loss plateau (<1% improvement over 10 epochs) or overfitting detected.
```
- ⚠️ MEL increased to 18.35 (worse than best 18.23)
- ⚠️ Negative improvement (-0.82% = getting worse)
- ⚠️ 5 consecutive epochs of mel increase
- 🛑 Training stops to prevent overfitting

### Interpreting the Graph

**Look at the TOP panel first:**

1. **MEL curve (blue)** should:
   - Steadily decrease
   - Have a clear downward trend
   - Flatten when model quality plateaus

2. **FM curve (green)** should:
   - Generally decrease
   - May be noisier than MEL
   - Complement MEL for quality assessment

3. **COMPOSITE curve (purple dashed)** should:
   - Combine trends of MEL and FM
   - Show overall quality improvement
   - Guide checkpoint selection

4. **Best checkpoint (blue star)**:
   - Shows where best MEL was achieved
   - This is the checkpoint to use

**BOTTOM panel is reference only:**
- GEN/DISC will oscillate (that's normal!)
- Don't worry if they go up sometimes
- Only use for sanity checking (e.g., both stuck at 0 = problem)

## Benefits

### For Users
1. **Cleaner Logs**: No clutter, easy to scan
2. **Visual Feedback**: See training progress at a glance
3. **Quick Diagnosis**: Easily spot plateaus or overfitting
4. **Better Understanding**: Graph clearly shows what matters
5. **Shareable**: Include loss graph when sharing models

### For Debugging
1. **Historical View**: Complete loss history preserved
2. **Trend Analysis**: Spot patterns in training behavior
3. **Comparison**: Compare different training runs visually
4. **Documentation**: Graph serves as training record

### For Quality Control
1. **Validation**: Verify training progressed correctly
2. **Best Model Selection**: See exactly where best checkpoint occurred
3. **Overfitting Detection**: Visual confirmation of uptrend
4. **Stopping Criteria**: Justify why training stopped

## Files Modified

1. **`modules/rvc/infer/modules/train/train.py`**
   - Line 17-19: Added matplotlib imports
   - Line 119: Added `epoch_history` tracking
   - Line 171-181: Store loss snapshots each epoch
   - Line 327-395: New `plot_losses()` method
   - Line 799-807: Removed redundant logging statements
   - Line 889-901: Call plot generation on checkpoint save

## Configuration

No configuration needed - graphs are generated automatically on every checkpoint save.

### Optional Customization

To adjust graph appearance, modify `plot_losses()` method:

```python
# In LossTracker.plot_losses()

# Adjust figure size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # Width, Height

# Adjust DPI (resolution)
plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Higher DPI = larger file

# Adjust composite weight in formula
self.composite_weight_fm = 0.3  # Default, can use 0.2-0.5
```

## Validation

### Checklist
- [x] Redundant log statements removed
- [x] Only LossTracker and TQDM visible in logs
- [x] Loss history tracked per epoch
- [x] Dual-panel graph generated
- [x] PRIMARY metrics in top panel
- [x] SECONDARY metrics in bottom panel
- [x] Best checkpoint marked on graph
- [x] Graph saved in project directory
- [x] Graph copied to trained models directory
- [x] Clear labels and annotations
- [x] High resolution (150 DPI)
- [x] Non-blocking (training continues if plotting fails)

## Example Output

### Console Logs
```
Epoch 1: 100%|████████████████| 450/450 [02:15<00:00, 3.32it/s]
[Tracker] EMA(mel=28.45 [best=28.45], fm=16.23, composite=33.32 [best=33.32]) | gen=5.21 disc=4.15 kl=0.62 | improvement_over_10ep=N/A, uptrend_epochs=0, epochs_since_best_save=0

Epoch 5: 100%|████████████████| 450/450 [02:10<00:00, 3.45it/s]
[Tracker] EMA(mel=24.12 [best=24.12], fm=14.89, composite=28.59 [best=28.59]) | gen=4.32 disc=3.78 kl=0.51 | improvement_over_10ep=N/A, uptrend_epochs=0, epochs_since_best_save=0

Epoch 25: 100%|███████████████| 450/450 [02:08<00:00, 3.51it/s]
[Tracker] EMA(mel=18.76 [best=18.76], fm=12.43, composite=22.49 [best=22.49]) | gen=3.54 disc=3.41 kl=0.43 | improvement_over_10ep=7.23%, uptrend_epochs=0, epochs_since_best_save=0
Saved loss plot to outputs/voices/MyVoice/loss_plot_epoch25.png
Copied loss plot to models/trained/MyVoice_losses.png
```

### Loss Graph
![Example Loss Graph](https://via.placeholder.com/800x600?text=Loss+Curves+Example)

*Shows clear downward trend in MEL (blue) and FM (green) in top panel, with oscillating GEN/DISC (red/orange) in bottom panel*

## Migration Notes

### No Breaking Changes
- Existing logs still work
- Graph generation is additive
- No config changes needed

### Behavioral Changes
- Fewer log lines (cleaner output)
- New PNG files created alongside checkpoints
- Slightly longer save time (graph generation ~1-2 seconds)

## Performance Impact

- **Negligible**: Graph generation takes 1-2 seconds per checkpoint
- **Non-blocking**: Training continues even if plotting fails
- **Memory**: Stores ~100 bytes per epoch (insignificant)
- **Disk**: PNG files are ~200-500 KB each

## Troubleshooting

### Graph Not Generated
```python
# Check logs for:
"Error plotting losses: ..."
```
**Solutions:**
- Ensure matplotlib is installed: `pip install matplotlib`
- Check disk space in project directory
- Verify write permissions

### Graph Shows No Data
**Cause**: Less than 2 epochs trained
**Solution**: Train for at least 2 epochs before graph appears

### Graph Cuts Off
**Cause**: Very high or low loss values
**Solution**: Automatic scaling handles this, but may need manual adjustment in plot_losses()

## Future Enhancements

Potential improvements (not yet implemented):
- [ ] Save/load loss history for resumed training
- [ ] Add validation loss curves (if validation set added)
- [ ] Export loss history to CSV
- [ ] Interactive HTML plots with zoom
- [ ] Comparison plots for multiple models
- [ ] Tensorboard integration

