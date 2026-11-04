# Credits

## Host voices

The Worth Telling hosts are voiced with Chatterbox Turbo (Resemble AI, MIT licence), cloned from three
speakers of LibriTTS-R:

| Host | LibriTTS-R speaker |
|---|---|
| Ren | 7021 |
| Bex | 174 |
| Ozzy | 2300 |

LibriTTS-R: Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han,
A. Bapna, "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", Interspeech 2023.
Licence CC BY 4.0, https://www.openslr.org/141/. It is derived from LibriTTS (H. Zen et al., 2019) and
LibriSpeech, built from public-domain LibriVox audiobook recordings.

The voice samples are in `profiles/worth_telling/voices/<host>/sample.wav`, unchanged except that a few
of the speaker's utterances are joined with short pauses.

## Other

- Depth model: MiDaS v2.1 small, MIT.
- Caption font: DejaVu Sans Bold, see assets/fonts/DejaVu-LICENSE.txt.
- Speech recognition used while choosing voices: Whisper small.en (MIT) through sherpa-onnx (Apache 2.0).
