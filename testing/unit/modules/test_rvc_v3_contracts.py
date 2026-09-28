"""Regression checks against real V2 topology; no model downloads required."""
import unittest
import torch
import tempfile
import json
from pathlib import Path

from modules.rvc_v3.configs.v3_config import RVCV3Config, get_default_config
from modules.rvc_v3.models.generator import TextConditionedTextEncoder
from modules.rvc.infer.lib.infer_pack.models import TextEncoder


class TestV3Contracts(unittest.TestCase):
    def test_sample_rate_contracts(self):
        for sr in (32000, 40000, 44100, 48000):
            get_default_config(sr).validate_contract()
        broken = get_default_config(48000)
        broken.hop_length = 512
        with self.assertRaises(ValueError): broken.validate_contract()

    def test_old_metadata_uses_legacy_backbone(self):
        self.assertEqual(RVCV3Config.from_dict({}).backbone, 'legacy')
        self.assertEqual(RVCV3Config.from_dict({}).text_tokenizer, 'legacy')

    def test_character_vocabulary_is_shared(self):
        from modules.rvc_v3.text_tokens import encode_char_v1
        from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
        from modules.rvc_v3.training.train_wrapper import SimplePhonemizer
        pipeline=RVCV3Pipeline.__new__(RVCV3Pipeline);pipeline.config=get_default_config(48000)
        self.assertEqual(pipeline._encode_lyrics_tokens('A quiet word')[0],encode_char_v1('A quiet word'))
        self.assertEqual(encode_char_v1('\u2603'),[1])
        self.assertEqual(SimplePhonemizer('char_v1').phonemize('A quiet word'),encode_char_v1('A quiet word'))

    def test_transferred_encoder_matches_v2_without_text(self):
        torch.manual_seed(4)
        source = TextEncoder(768, 16, 16, 32, 2, 2, 3, 0).eval()
        dest = TextConditionedTextEncoder(16,16,32,2,2,3,0,
                                         text_d_model=16, backbone='v2_compatible').eval()
        dest.base_encoder.load_state_dict(source.state_dict(), strict=True)
        phone, pitch, lengths = torch.randn(2,20,768), torch.ones(2,20,dtype=torch.long), torch.tensor([20,13])
        with torch.no_grad():
            expected = source(phone,pitch,lengths)
            actual = dest(phone,pitch,lengths,text_features=torch.randn(2,4,16),text_strength=0)
        for a,b in zip(expected,actual): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_missing_text_row_is_finite_and_unchanged(self):
        from modules.rvc_v3.models.text_encoder import TextEncoder as LyricsEncoder
        text=LyricsEncoder(200,16,2,1,32,0).eval()
        mask=torch.tensor([[False,False],[True,True]])
        features=text(torch.ones(2,2,dtype=torch.long),mask)
        self.assertTrue(torch.isfinite(features).all())
        model=TextConditionedTextEncoder(16,16,32,2,2,3,0,text_d_model=16,backbone='v2_compatible').eval()
        phone=torch.randn(2,10,768);pitch=torch.ones(2,10,dtype=torch.long);lengths=torch.tensor([10,7])
        base=model(phone,pitch,lengths)
        conditioned=model(phone,pitch,lengths,text_features=features,text_mask=mask)
        self.assertTrue(torch.isfinite(conditioned[0]).all())
        torch.testing.assert_close(base[0][1],conditioned[0][1],rtol=0,atol=0)
        self.assertEqual(conditioned[0][1,:,7:].abs().sum().item(),0)

    def test_lyrics_follow_file_and_crop(self):
        from modules.rvc_v3.training.dataset import RVCV3Dataset
        class Phonemes:
            def phonemize(self,text,**kwargs): return text.split()
            def encode(self,tokens): return [dict(early=1,late=2)[x] for x in tokens]
        with tempfile.TemporaryDirectory() as tmp:
            data=RVCV3Dataset.__new__(RVCV3Dataset)
            data.config=RVCV3Config.from_dict({})
            data.lyrics_dir=Path(tmp);data.phonemizer=Phonemes()
            data.lyrics_data={'segments':[dict(text='early',start=0,end=1)]}
            self.assertTrue(data._get_text_tokens('song',0,2)[1].all())
            (Path(tmp)/'song.json').write_text(json.dumps({'segments':[dict(text='early',start=0,end=1),dict(text='late',start=4,end=5)]}))
            self.assertEqual(data._get_text_tokens('song',3,6)[0].tolist(),[2])


if __name__ == '__main__': unittest.main()
