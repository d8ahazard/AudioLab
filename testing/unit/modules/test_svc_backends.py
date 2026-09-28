"""Capability boundaries must fail before model download or expensive inference."""
import unittest
from modules.svc_backends import convert

class TestSingingBackends(unittest.TestCase):
    def test_unsupported_delivery_is_explicit(self):
        for backend in ('seed_vc','yingmusic'):
            with self.assertRaisesRegex(ValueError,'requires Vevo2'):
                convert('missing','missing','unused',backend=backend,mode='target_style')

    def test_target_delivery_needs_transcripts(self):
        with self.assertRaisesRegex(ValueError,'source lyrics and reference transcript'):
            convert('missing','missing','unused',backend='vevo2',mode='target_style')

    def test_lyrics_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'does not accept lyric'):
            convert('missing','missing','unused',backend='seed_vc',lyrics='words')

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown conversion mode'):
            convert('missing','missing','unused',mode='random')

if __name__=='__main__':unittest.main()
