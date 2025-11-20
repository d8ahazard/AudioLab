# RVC Training Quick Start Guide

## TL;DR - What You Need to Know

### The One Thing to Remember
**Watch the MEL loss (blue line).** Everything else is secondary.

### Model Version Selection

- **V1**: Legacy 256-dim features (40k/48k only)
- **V2**: Standard 768-dim features (32k/40k/48k) - Most commonly used
- **V3**: Enhanced architecture with text conditioning - Better quality, requires pretrained models
  - Automatically uses V2 weights as initialization
  - Larger model with cross-attention
  - Recommended for 48k sample rate
  - Best quality but requires more VRAM

### Quick Status Check During Training
```
[Tracker] EMA(mel=22.38 [best=21.50], ...)
          ^^^^^^         ^^^^^^^^^^^
          Current        Best so far
```

- **If current ≈ best**: Training is progressing well ✅
- **If current > best for many epochs**: Might be overfitting ⚠️
- **If `improvement_over_10ep` < 1%**: Training will stop soon (plateau) 🛑

## Reading the Logs

### Example Log Line
```
[Tracker] EMA(mel=22.38 [best=21.50], fm=13.43, composite=26.41 [best=25.93]) | 
         gen=3.51 disc=3.40 kl=0.45 | improvement_over_10ep=5.12%, 
         uptrend_epochs=0, epochs_since_best_save=3
```

### What to Watch

| Metric | Good | Warning | Stop |
|--------|------|---------|------|
| `mel` | Decreasing | Stable | Increasing |
| `improvement_over_10ep` | > 3% | 1-2% | < 1% |
| `uptrend_epochs` | 0-2 | 3-4 | ≥ 5 |
| `epochs_since_best_save` | 0-5 | 6-10 | > 10 |

### What to Ignore

- `gen` - Will oscillate, this is normal
- `disc` - Will oscillate, this is normal
- `kl` - Just monitoring, don't worry about it

## Reading the Graph    

### Find Your Graph
After training, look for:
```
models/trained/{your_model_name}_losses.png
```

### What You'll See

**TOP PANEL (Primary Metrics)**
- 📘 **Blue line (MEL)**: Should go down steadily
- 📗 **Green line (FM)**: Should generally decrease
- 🟣 **Purple dashed (COMPOSITE)**: Combines blue + green
- ⭐ **Blue star**: Marks the best checkpoint (use this one!)

**BOTTOM PANEL (Secondary Metrics)**
- 🔴 **Red (Generator)**: Will zigzag - ignore it
- 🟠 **Orange (Discriminator)**: Will zigzag - ignore it
- 🟤 **Brown (KL)**: Just monitoring

### Good vs Bad Training

**✅ Good Training:**
```
MEL (blue) consistently going down
No sharp upward spikes
Star near the end
```

**⚠️ Plateau:**
```
MEL (blue) flattening out
No improvement for many epochs
Star far from the end
```

**❌ Overfitting:**
```
MEL (blue) starting to go UP
Quality degrading
Star in the middle, not at end
```

## Common Scenarios

### Scenario 1: "Is my training working?"

**Check:**
1. Look at the `[Tracker]` line
2. Is `mel` going down? ✅ Yes → Working well
3. Is `improvement_over_10ep` > 1%? ✅ Yes → Still improving

**Action:** Keep training

---

### Scenario 2: "Training stopped early, why?"

**Check the last log line:**

```
[Tracker] Early stopping: mel loss plateau (<1% improvement over 10 epochs)
```

**Reason:** Model quality stopped improving
**Action:** This is normal! Use the best checkpoint (marked with ⭐ in graph)

---

### Scenario 3: "Which checkpoint should I use?"

**Answer:** Use the one marked with ⭐ in the loss graph

**Or:** Look for the `[best=X.XX]` value in logs - use the checkpoint from when mel equaled that value

---

### Scenario 4: "My logs look different than before"

**Old logs (verbose, messy):**
```
Train Epoch: 74 [50%]
[1234, 0.0001]
loss_disc=3.510, loss_gen=3.510, loss_fm=13.430, loss_mel=22.380, loss_kl=0.450
[Tracker] EMA(gen=4.19 disc=3.51 ...)
```

**New logs (clean, focused):**
```
Epoch 74: 100%|████████████| 450/450 [02:15<00:00, 3.32it/s]
[Tracker] EMA(mel=22.38 [best=21.50], fm=13.43, composite=26.41 [best=25.93]) | ...
```

**Reason:** Cleaned up logging for clarity
**Action:** Enjoy the cleaner logs! 😊

---

### Scenario 5: "Generator/Discriminator losses are jumping around"

**This is normal!**

Gen and disc losses naturally oscillate in GAN training. They're playing a competitive game, so they go up and down.

**Don't worry about it.** Focus on MEL (blue line) instead.

---

### Scenario 6: "When will training finish?"

**It depends on:**
- Model quality improvement rate
- Plateau detection (< 1% improvement over 10 epochs)
- Overfitting detection (5+ consecutive epochs of mel increase)

**Typically:**
- Small dataset (< 30 min audio): 50-150 epochs
- Medium dataset (30-60 min audio): 100-300 epochs
- Large dataset (> 60 min audio): 200-500 epochs

**Training stops automatically when optimal.**

## Decision Tree

```
Start Training
     |
     v
  [Check mel]
     |
     ├─> Going down? ──> ✅ Keep training
     |
     ├─> Flat for 10+ epochs? ──> 🛑 Stops automatically (plateau)
     |
     └─> Going up for 5+ epochs? ──> 🛑 Stops automatically (overfitting)
```

## Configuration (Advanced)

### Default Settings
```python
# In the code (modules/rvc/infer/modules/train/train.py)
significant_improvement_threshold = 0.01  # 1% improvement required
plateau_patience_epochs = 10              # 10 epoch window
composite_weight_fm = 0.3                 # MEL + 0.3×FM
```

### If Training Stops Too Early
```python
# Make it more patient
significant_improvement_threshold = 0.005  # 0.5% improvement required
plateau_patience_epochs = 15               # 15 epoch window
```

### If Training Takes Too Long
```python
# Make it less patient
significant_improvement_threshold = 0.02   # 2% improvement required
plateau_patience_epochs = 8                # 8 epoch window
```

## FAQ

**Q: Why did my training stop at epoch 85 when I set 300 epochs?**
A: Because the model quality plateaued (< 1% improvement over 10 epochs). The best checkpoint was already saved. Continuing would waste time without improving quality.

**Q: Should I be worried if gen/disc losses are oscillating?**
A: No! This is completely normal for GAN training. Focus on MEL loss instead.

**Q: What if MEL loss is 20 but the graph shows best at 18?**
A: The model is overfitting. Use the checkpoint from when MEL was 18 (marked with ⭐).

**Q: How do I know which checkpoint file to use?**
A: Look for the file with the epoch number closest to where the ⭐ appears in the graph. Or use the final `{model_name}.pth` file which is automatically the best one.

**Q: Can I ignore the graph and just look at logs?**
A: Yes! The `[Tracker]` line in logs tells you everything. The graph is just a visual aid.

**Q: What's the difference between mel, fm, and composite?**
A:
- **MEL**: Audio reconstruction quality (primary)
- **FM**: Naturalness/smoothness (secondary)
- **COMPOSITE**: Combined metric (mel + 0.3×fm)

All three should go down. MEL is most important.

**Q: My old checkpoints from before this update - are they still good?**
A: Yes! This update only changed how training decides when to stop and save. Old checkpoints work perfectly.

## Cheat Sheet

### Good Signs ✅
- ✅ MEL decreasing steadily
- ✅ `improvement_over_10ep` > 3%
- ✅ `uptrend_epochs = 0`
- ✅ `[best=X.XX]` updating frequently
- ✅ Graph shows steady downward trend

### Warning Signs ⚠️
- ⚠️ MEL not changing much
- ⚠️ `improvement_over_10ep` 1-2%
- ⚠️ `uptrend_epochs = 3-4`
- ⚠️ `epochs_since_best_save > 10`
- ⚠️ Graph shows flattening

### Stop Signs 🛑
- 🛑 MEL increasing
- 🛑 `improvement_over_10ep < 1%`
- 🛑 `uptrend_epochs ≥ 5`
- 🛑 Training auto-stops
- 🛑 Graph shows upward trend at end

## Still Confused?

### The Simplest Explanation

**Before (wrong):**
"Training used generator loss to decide when to save/stop, but generator loss oscillates and doesn't indicate quality."

**After (correct):**
"Training now uses MEL loss (audio reconstruction quality) to decide when to save/stop."

**What you do:**
Nothing different! Just look at cleaner logs and get a nice graph.

**What changes:**
Training stops automatically at the optimal point instead of wasting epochs.

---

## Contact

If you encounter issues, check:
1. The loss graph - does it show reasonable trends?
2. The last few `[Tracker]` log lines - what do they say?
3. The early stopping message - why did it stop?

Most "issues" are actually the system working correctly (stopping at optimal point).

**Remember:** The goal is the best model quality, not the most epochs!

