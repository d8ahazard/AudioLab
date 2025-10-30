"""
Lyric editor and style tag annotator for RVC V3.

Provides data structures and utilities for editing lyrics with style tags.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class LyricSegment:
    """
    A single segment of lyrics with timing and optional style tags.
    
    Attributes:
        start: Start time in seconds
        end: End time in seconds
        text: The lyric text
        tags: List of style tags (e.g., ['clean'], ['raspy', 'breathy'])
        confidence: Confidence score from transcription (0-1)
    """
    start: float
    end: float
    text: str
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def duration(self) -> float:
        """Get the duration of this segment."""
        return self.end - self.start
    
    def add_tag(self, tag: str):
        """Add a style tag to this segment."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove a style tag from this segment."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if segment has a specific tag."""
        return tag in self.tags
    
    def get_tagged_text(self) -> str:
        """Get text with tags in square brackets."""
        if not self.tags:
            return self.text
        tag_str = ''.join([f'[{tag}]' for tag in self.tags])
        return f"{tag_str} {self.text}"


class LyricEditor:
    """
    Editor for lyrics with style tags.
    
    Manages a collection of lyric segments and provides utilities for
    editing, tagging, and exporting to training format.
    """
    
    # Common style tags
    COMMON_TAGS = [
        'clean',
        'raspy',
        'breathy',
        'distorted',
        'nasal',
        'soft',
        'belted',
        'whisper',
        'growl',
        'vibrato',
        'straight',
        'head_voice',
        'chest_voice',
        'falsetto',
    ]
    
    def __init__(self, project_name: str, output_dir: str):
        """
        Initialize the lyric editor.
        
        Args:
            project_name: Name of the project
            output_dir: Base output directory
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.lyrics_dir = self.output_dir / project_name / "lyrics"
        self.lyrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.segments: List[LyricSegment] = []
        self.metadata: Dict = {}
    
    def load_from_transcript(self, transcript: Dict) -> int:
        """
        Load segments from a transcript.
        
        Args:
            transcript: Transcript dict from Transcriber
        
        Returns:
            Number of segments loaded
        """
        self.segments = []
        
        for seg in transcript.get('segments', []):
            segment = LyricSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'],
                confidence=seg.get('confidence', 1.0)
            )
            self.segments.append(segment)
        
        self.metadata = {
            'language': transcript.get('language', 'unknown'),
            'full_text': transcript.get('full_text', '')
        }
        
        logger.info(f"Loaded {len(self.segments)} segments from transcript")
        return len(self.segments)
    
    def load_from_file(self, file_path: str) -> int:
        """Load annotated lyrics from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.segments = []
        for seg_data in data.get('segments', []):
            segment = LyricSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                text=seg_data['text'],
                tags=seg_data.get('tags', []),
                confidence=seg_data.get('confidence', 1.0)
            )
            self.segments.append(segment)
        
        self.metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(self.segments)} segments from {file_path}")
        return len(self.segments)
    
    def save(self, filename: str = "annotated_lyrics.json"):
        """Save annotated lyrics to a JSON file."""
        output_file = self.lyrics_dir / filename
        
        data = {
            'project_name': self.project_name,
            'metadata': self.metadata,
            'segments': [asdict(seg) for seg in self.segments]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.segments)} segments to {output_file}")
        return str(output_file)
    
    def add_segment(self, segment: LyricSegment):
        """Add a segment to the lyrics."""
        self.segments.append(segment)
        # Keep segments sorted by start time
        self.segments.sort(key=lambda s: s.start)
    
    def remove_segment(self, index: int):
        """Remove a segment by index."""
        if 0 <= index < len(self.segments):
            self.segments.pop(index)
    
    def merge_segments(self, start_idx: int, end_idx: int) -> Optional[LyricSegment]:
        """
        Merge multiple segments into one.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
        
        Returns:
            The merged segment or None if invalid indices
        """
        if not (0 <= start_idx <= end_idx < len(self.segments)):
            return None
        
        segments_to_merge = self.segments[start_idx:end_idx+1]
        
        merged = LyricSegment(
            start=segments_to_merge[0].start,
            end=segments_to_merge[-1].end,
            text=' '.join([s.text for s in segments_to_merge]),
            tags=segments_to_merge[0].tags.copy(),  # Use first segment's tags
            confidence=min([s.confidence for s in segments_to_merge])
        )
        
        # Remove old segments and insert merged one
        self.segments = (
            self.segments[:start_idx] +
            [merged] +
            self.segments[end_idx+1:]
        )
        
        return merged
    
    def split_segment(self, index: int, split_time: float) -> bool:
        """
        Split a segment at a specific time.
        
        Args:
            index: Index of segment to split
            split_time: Time to split at
        
        Returns:
            True if split was successful
        """
        if not (0 <= index < len(self.segments)):
            return False
        
        segment = self.segments[index]
        
        if not (segment.start < split_time < segment.end):
            return False
        
        # Create two new segments
        seg1 = LyricSegment(
            start=segment.start,
            end=split_time,
            text=segment.text,  # User will need to edit text
            tags=segment.tags.copy(),
            confidence=segment.confidence
        )
        
        seg2 = LyricSegment(
            start=split_time,
            end=segment.end,
            text="",  # User will need to add text
            tags=segment.tags.copy(),
            confidence=segment.confidence
        )
        
        # Replace original with split segments
        self.segments = (
            self.segments[:index] +
            [seg1, seg2] +
            self.segments[index+1:]
        )
        
        return True
    
    def apply_tag_to_range(self, start_idx: int, end_idx: int, tag: str):
        """Apply a tag to a range of segments."""
        for i in range(start_idx, min(end_idx + 1, len(self.segments))):
            self.segments[i].add_tag(tag)
    
    def remove_tag_from_range(self, start_idx: int, end_idx: int, tag: str):
        """Remove a tag from a range of segments."""
        for i in range(start_idx, min(end_idx + 1, len(self.segments))):
            self.segments[i].remove_tag(tag)
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags used in the lyrics."""
        all_tags = set()
        for segment in self.segments:
            all_tags.update(segment.tags)
        return sorted(list(all_tags))
    
    def export_for_training(self, output_format: str = "tagged_text") -> str:
        """
        Export lyrics in format suitable for training.
        
        Args:
            output_format: Format to export ('tagged_text', 'json', 'tokens')
        
        Returns:
            Path to exported file
        """
        if output_format == "tagged_text":
            # Export as text with inline tags
            output_file = self.lyrics_dir / "lyrics_tagged.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for segment in self.segments:
                    f.write(f"{segment.start:.2f}-{segment.end:.2f}: {segment.get_tagged_text()}\n")
        
        elif output_format == "json":
            # Export as JSON (same as save)
            output_file = self.lyrics_dir / "lyrics_training.json"
            data = {
                'segments': [asdict(seg) for seg in self.segments],
                'all_tags': self.get_all_tags()
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Exported lyrics to {output_file}")
        return str(output_file)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the lyrics."""
        total_duration = sum(seg.duration() for seg in self.segments)
        tagged_segments = sum(1 for seg in self.segments if seg.tags)
        
        return {
            'num_segments': len(self.segments),
            'total_duration': total_duration,
            'tagged_segments': tagged_segments,
            'tag_coverage': tagged_segments / len(self.segments) if self.segments else 0,
            'unique_tags': len(self.get_all_tags()),
            'all_tags': self.get_all_tags()
        }

