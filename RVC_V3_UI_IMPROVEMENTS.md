# RVC V3 UI Improvements - Batch Processing & Lyric Editing

## ✅ Major Improvements Made

### 1. **Batch URL Download** (Like Existing RVC Interface)
**Before**: Single URL input, one song at a time
**After**: Multiline textbox supporting multiple URLs

```python
song_urls = gr.Textbox(
    label="Audio URLs",
    placeholder="https://youtube.com/watch?v=...\nhttps://youtube.com/watch?v=...\n(one per line)",
    lines=5
)
```

- ✅ Enter multiple YouTube URLs (one per line)
- ✅ Downloads all songs in batch
- ✅ Creates separate project for each song (`project_name_song_01`, `project_name_song_02`, etc.)
- ✅ Extracts metadata and titles

### 2. **Song Selector Dropdown**
**New**: Select individual songs for editing/processing

```python
song_selector = gr.Dropdown(
    label="Select Song to Process",
    choices=[],  # Populated after download
    interactive=True
)
```

- ✅ Lists all songs with titles: `"project_song_01: Never Gonna Give You Up"`
- ✅ Auto-updates after download
- ✅ Manual refresh button available
- ✅ Allows individual song processing

### 3. **Per-Song Lyric Editor**
**New**: Full lyric editing interface with style tag support

```python
lyrics_editor = gr.Textbox(
    label="Lyrics Editor (add tags like: [clean] your lyrics here)",
    lines=15,
    interactive=True
)
```

**Features**:
- ✅ Load lyrics for selected song
- ✅ Edit transcript text
- ✅ Add style tags inline: `[clean]`, `[raspy]`, `[breathy]`, etc.
- ✅ Save annotated lyrics with tags
- ✅ Preserves timing information from transcription

**Example Usage**:
```
[clean] Never gonna give you up
[raspy] Never gonna let you down
[belted] Never gonna run around and desert you
```

### 4. **Batch Processing**
**New**: Process all songs at once

```python
batch_process_btn = gr.Button("🚀 Batch Process All Songs", variant="primary")
```

- ✅ Separates vocals for all songs
- ✅ Transcribes all songs
- ✅ Shows progress per song
- ✅ Reports success/failure for each

### 5. **Enhanced Feature Extraction**
**Updated**: Works across all songs in project

- ✅ Extracts features from all downloaded songs
- ✅ Combines features for unified model training
- ✅ Progress tracking per song

### 6. **Unified Index Building**
**Updated**: Builds single index from all songs

- ✅ Collects features from all song projects
- ✅ Creates single retrieval index for the voice model
- ✅ Reports total feature count

## 🎯 Complete Workflow

### Recommended Workflow:

1. **Create Project**: Enter project name (e.g., "taylor_swift")

2. **Download Songs**: 
   - Paste multiple YouTube URLs (one per line)
   - Click "Download All"
   - Songs are named: `taylor_swift_song_01`, `taylor_swift_song_02`, etc.

3. **Batch Process** (Optional):
   - Click "🚀 Batch Process All Songs"
   - Automatically separates and transcribes all songs
   
   OR process individually:
   - Select song from dropdown
   - Click "Separate Vocals"
   - Click "Transcribe Lyrics"

4. **Edit Lyrics Per Song**:
   - Select song from dropdown
   - Click "Load Lyrics for Editing"
   - Edit text and add style tags
   - Click "Save Edited Lyrics"
   - Repeat for each song

5. **Extract Features**:
   - Click "Extract Features" (processes all songs)
   - Extracts HuBERT + Whisper features

6. **Build Index**:
   - Click "Build Index" (combines all songs)
   - Creates unified retrieval index

7. **Train Model**:
   - Configure epochs and batch size
   - Click "Start Training"
   - Trains on all annotated songs

8. **Inference**:
   - Load trained model
   - Convert new audio with style tags

## 📋 UI Layout Structure

### Data Preparation Tab

```
Step 1: Download Songs
├─ Audio URLs (multiline textbox)
└─ Download All button

Step 2: Select Song & Process
├─ Song Selector (dropdown)
├─ Refresh Song List button
├─ Separate Vocals button (for selected song)
├─ Transcribe Lyrics button (for selected song)
├─ Batch Process All Songs button (for all songs)
└─ Batch Processing Status

Step 3: Edit Lyrics & Add Style Tags
├─ Load Lyrics for Editing button
├─ Lyrics Editor (multiline textbox with tags)
├─ Save Edited Lyrics button
└─ Style Tags reference
```

### Training Tab (Updated)

```
Feature Extraction
├─ Processes ALL songs in project
└─ Combined feature cache

Build Retrieval Index
├─ Collects features from ALL songs
└─ Single unified index

Start Training
├─ Trains on ALL annotated songs
└─ Single voice model
```

### Inference Tab

```
Convert Audio
├─ Uses trained model from all songs
└─ Apply style tags during conversion
```

## 🔧 Technical Implementation

### Song Organization

```
outputs/rvc_v3_data/
└── my_voice_project/
    ├── retrieval_index.index          (unified index)
    ├── retrieval_index.npy            (unified features)
    ├── checkpoints/                    (trained models)
    └── (parent project for UI reference)

outputs/rvc_v3_data/my_voice_project_song_01/
├── raw/
│   ├── Song Title.wav
│   └── Song Title.info.json
├── vocals/
│   └── Vocals.wav
├── lyrics/
│   ├── transcript.json              (raw transcription)
│   └── annotated_lyrics.json        (with tags)
└── features/
    └── *.pt (cached features)

outputs/rvc_v3_data/my_voice_project_song_02/
└── (same structure)
```

### Key Functions Added

1. `download_songs()` - Batch download from multiple URLs
2. `get_song_project_name()` - Extract project from selector string
3. `separate_selected_song()` - Process individual song
4. `transcribe_selected_song()` - Transcribe individual song
5. `load_lyrics_for_editing()` - Load lyrics for editing
6. `save_edited_lyrics()` - Save with tags
7. `get_all_songs_in_project()` - List all songs
8. `batch_process_all_songs()` - Batch separate + transcribe
9. Updated `extract_features()` - Process all songs
10. Updated `build_index()` - Combine all features

## ✨ User Experience Improvements

### Before (Original Design):
- ❌ Single URL download only
- ❌ No per-song management
- ❌ Limited lyric editing
- ❌ Manual processing for each song

### After (Enhanced Design):
- ✅ Batch URL download (like existing RVC)
- ✅ Song-by-song selection and editing
- ✅ Full lyric editor with inline tag support
- ✅ Batch processing option
- ✅ Clear workflow steps
- ✅ Better organization

## 🎉 Benefits

1. **Efficient Training Data Preparation**:
   - Download 10+ songs at once
   - Batch process them automatically
   - Edit lyrics individually for accuracy

2. **Fine-Grained Control**:
   - Review and edit each song's lyrics
   - Add different style tags per song
   - Ensure transcription accuracy

3. **Professional Workflow**:
   - Matches existing RVC interface patterns
   - Familiar UX for AudioLab users
   - Streamlined data preparation

4. **Better Voice Models**:
   - More training data from multiple songs
   - Accurate lyrics improve model quality
   - Style tags enable creative control

## 🚀 Ready to Use!

The enhanced RVC V3 UI is now:
- ✅ **Fully functional**
- ✅ **Batch-processing capable**
- ✅ **Per-song lyric editing**
- ✅ **Integrated with main AudioLab UI**
- ✅ **Tested and validated**

**Start AudioLab and navigate to the RVC V3 tab to begin!** 🎤✨

