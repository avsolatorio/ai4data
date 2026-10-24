---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The accessibility, caption, metadata, loudness, and statistical standards each chapter builds on, with the program's own tool and guide, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Each entry below
was checked against its source in October 2026, and the version read is
named.

## Accessibility of video

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Web Content Accessibility Guidelines (WCAG) 2.2](https://www.w3.org/TR/WCAG22/), W3C Recommendation of 12 December 2024 | Success criteria 1.2.2 Captions (Prerecorded, level A), 1.2.3 Audio Description or Media Alternative (Prerecorded, A), 1.2.5 Audio Description (Prerecorded, AA), 1.2.8 Media Alternative (Prerecorded, AAA), 1.4.2 Audio Control (A), and 2.3.1 Three Flashes or Below Threshold (A) | | Chapters [6](./review.mdx) and [7](./publishing.mdx) |
| [Understanding success criterion 1.2.5](https://www.w3.org/WAI/WCAG22/Understanding/audio-description-prerecorded.html), W3C | The statement that no additional audio description is needed where all of the video information is already provided in the existing audio | | Chapters [3](./narration.mdx) and 6 |
| [WCAG 2.2, success criterion 1.4.7 Low or No Background Audio](https://www.w3.org/TR/WCAG22/#low-or-no-background-audio) (AAA) | Background sounds at least 20 dB below speech, a criterion for prerecorded audio-only content; it applies to a podcast version of the narration and is a reference point for the music level of a video | | Chapter [5](./voice.mdx) |

## Captions and subtitles

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [WebVTT: The Web Video Text Tracks Format](https://www.w3.org/TR/webvtt1/), W3C Candidate Recommendation Draft of 20 May 2026 | The caption file format, the `WEBVTT` header, and the timing rules (a cue ends after it starts; no cue starts before the start of an earlier cue) | The program's video tool writes its captions in WebVTT | Chapter 6 |
| [HTML Living Standard, the track element](https://html.spec.whatwg.org/multipage/media.html#the-track-element), WHATWG | The `track` element of kind `captions` that attaches a WebVTT file to a `video` element | | Chapter 7 |
| [BBC Subtitle Guidelines](https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles/), version 1.2.5, March 2026 | Line length of 37 characters in Teletext and a share of the frame width online, at most two lines for landscape video, and a rate of 160 to 180 words per minute | | Chapter 6, default limits of `check_captions.py` |
| [Netflix English (USA) Timed Text Style Guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide) | 42 characters per line, two lines, and a reading speed of up to 20 characters per second for adult programmes | | Chapter 6, default reading speed of `check_captions.py` |

## Catalogue records

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [schema.org VideoObject](https://schema.org/VideoObject) | The type of the catalogue record, with `duration`, `uploadDate`, `contentUrl`, `embedUrl`, `caption`, `transcript`, `isBasedOn`, and `license` | The program's cookbooks use schema.org records for datasets | Chapter 7, `video_record.py` |
| [schema.org digitalSourceType](https://schema.org/digitalSourceType) and the [IPTC digital source type vocabulary](https://cv.iptc.org/newscodes/digitalsourcetype/) | The value `CompositeSyntheticDigitalSource`, defined by IPTC as a mix or composite of several elements of which at least one is generative AI | | Chapters 5 and 7 |
| [Schema.org Accessibility Properties for Discoverability Vocabulary](https://w3c-cg.github.io/a11y-discov-vocab/latest/), version 1.1, W3C Community Group report of 24 August 2026 | The values `closedCaptions` and `transcript` of `accessibilityFeature` (the older value `captions` is deprecated) and `auditory` and `visual` of `accessMode` | | Chapter 7 |
| [World Bank metadata schemas](https://worldbank.github.io/metadata-schemas/) | The video schema (`video-schema.json`), based on Dublin Core and schema.org VideoObject, for describing videos in the organization's catalogue | Used by the World Bank's catalogs and the Metadata Editor | Chapter 7 |

## Audio

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Recommendation ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770), version 5, November 2023 | The algorithms that measure programme loudness (LUFS) and true-peak level | | Chapter 5 |
| [EBU R 128](https://tech.ebu.ch/publications/r128), version 5.0, November 2023 | Loudness normalization for broadcast at a target of -23 LUFS | The program's tool normalizes its narration with the `loudnorm` filter of ffmpeg, which implements EBU R 128 normalization, at a target of -16 LUFS | Chapter 5 |
| [Speech Synthesis Markup Language (SSML) 1.1](https://www.w3.org/TR/speech-synthesis11/), W3C Recommendation of 7 September 2010 | Markup for spoken forms of numbers and dates, an alternative to the `speak` field for a voice that supports it | | Chapter 3 |

## Statistical definitions

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Resolution concerning statistics of work, employment and labour underutilization](https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@stat/documents/normativeinstrument/wcms_230304.pdf), 19th International Conference of Labour Statisticians (2013), as amended by the 21st (2023) | The definitions of persons in employment (paragraph 27) and in unemployment (paragraph 47) that the running example's narration states | | Chapters [2](./planning.mdx) and 3 |

## Licences and regulation

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | The licence of the Kokoro text-to-speech model and of Playwright | The program's tool uses both | Chapters [4](./visuals.mdx) and 5 |
| [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), Article 50 | Transparency obligations for AI systems that generate synthetic audio and video, and for those who deploy them, where the regulation applies | | Chapter 5 |

## Program methods and tools

| Resource | Use | In this guide |
|---|---|---|
| [Video tool and production guide](https://github.com/worldbank/ai4data/tree/main/scripts/videos) | The step script format, the stage kit, the recorder (Playwright), the contact sheets, the voice (Kokoro `af_heart`), the synthesised music, and the WebVTT captions of the program's explainers | All chapters |
| [Proof-Carrying Numbers](/docs/pcn/) | The method of the program's first explainer video | Chapters [1](./topics.mdx) and 3 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
