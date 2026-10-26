"""Publish a recorded video to YouTube and write its entry in the site's video catalogue.

Each published video has an upload file next to its step script,
videos/<name>.youtube.json, with the title, the description, the chapters,
the tags, the playlist, the YouTube settings and the catalogue entry (see
README.md, "Publishing"). The video, the subtitles and the thumbnail are the
recorder's outputs: out/<name>.mp4, out/<name>.vtt and the page named in
"thumbnail", rendered by thumbnail.py.

Without options, the script checks the upload file and the files, and writes
the description it would send to out/<name>.youtube.txt. It sends nothing:

    .venv/bin/python publish.py videos/pcn-explainer-v2.youtube.json

--upload sends the video, the English subtitles and the thumbnail, adds the
video to its playlist, and writes the YouTube id and the month back into the
upload file. It needs an OAuth token for the channel, made once with
authorize.py, at ~/.config/ai4data/youtube-token.json (or $YOUTUBE_TOKEN).
A video uploaded through a Google Cloud project that has not passed the
YouTube API audit stays private until the project passes it; switch it to
unlisted or public in YouTube Studio.

--catalogue PATH writes or replaces the video's entry in the catalogue module
(website/src/content/videos.js), with the YouTube id, the length, and the
transcript taken from the narration of the step script.
"""
import argparse
import datetime
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = Path(os.environ.get("YOUTUBE_TOKEN", Path.home() / ".config" / "ai4data" / "youtube-token.json"))
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl"]
TIMING = re.compile(r"^(\d+):(\d\d):(\d\d)\.\d+ --> ")


def seconds_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def clock(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def cue_starts(vtt):
    """(start in seconds, text) for each cue of a WebVTT file."""
    out = []
    for block in re.split(r"\n\s*\n", vtt.read_text(encoding="utf-8"))[1:]:
        rows = block.strip("\n").split("\n")
        m = TIMING.match(rows[0]) if rows else None
        if m:
            out.append((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), " ".join(rows[1:])))
    return out


def chapters(spec, vtt, length, problems, notes):
    """Chapter times from the first cue that starts with each chapter's phrase.

    YouTube shows chapters when the first starts at 0:00, there are at least
    three, and each lasts at least ten seconds. A last chapter shorter than
    that is merged into the one before it.
    """
    cues = cue_starts(vtt)
    marks = []
    for title, phrase in spec["chapters"]:
        hit = next((t for t, text in cues if text.startswith(phrase)), None)
        if hit is None:
            problems.append(f'chapter "{title}": no subtitle cue starts with "{phrase}"')
            continue
        marks.append([hit, title])
    if not marks:
        return []
    marks[0][0] = 0
    if len(marks) > 1 and length - marks[-1][0] < 10:
        notes.append(f'chapter "{marks[-1][1]}" lasts under ten seconds and is merged into "{marks[-2][1]}"')
        marks.pop()
    for (a, title), (b, _) in itertools.pairwise(marks):
        if b - a < 10:
            problems.append(f'chapter "{title}" lasts {b - a} s; YouTube needs at least 10')
    if len(marks) < 3:
        problems.append("fewer than three chapters; YouTube shows none")
    return marks


def description(spec, marks):
    parts = ["\n\n".join(spec["description"])]
    if marks:
        parts.append("Chapters\n" + "\n".join(f"{clock(t)} {title}" for t, title in marks))
    if spec.get("links"):
        parts.append("Links\n" + "\n".join(f"{label}: {url}" for label, url in spec["links"]))
    if spec.get("about"):
        parts.append("About this video\n" + "\n".join(spec["about"]))
    return "\n\n".join(parts)


def check(spec, text, files, problems):
    if len(spec["title"]) > 100:
        problems.append(f"title has {len(spec['title'])} characters; the limit is 100")
    if len(text.encode("utf-8")) > 5000:
        problems.append(f"description has {len(text.encode('utf-8'))} bytes; the limit is 5000")
    for name, value in (("title", spec["title"]), ("description", text)):
        if "<" in value or ">" in value:
            problems.append(f"the {name} contains < or >, which YouTube rejects")
    if sum(len(t) + 2 * (" " in t) for t in spec.get("tags", [])) + max(len(spec.get("tags", [])) - 1, 0) > 500:
        problems.append("tags exceed 500 characters")
    for label, path in files.items():
        if not path.exists():
            problems.append(f"{label} not found: {path.relative_to(HERE)}")


def transcript(script):
    """The narration of a step script, one paragraph per scene."""
    paras, last = [], None
    for step in json.loads(script.read_text(encoding="utf-8"))["steps"]:
        if not step.get("say"):
            continue
        m = re.match(r"go\(\s*'?(\w+)'?\s*,", step.get("js", ""))
        scene = m.group(1) if m else last
        if scene != last:
            paras.append([])
            last = scene
        paras[-1].append(step["say"])
    return [" ".join(p) for p in paras]


def js(value):
    """A JavaScript literal for a string, in the single quotes of the catalogue module."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def catalogue_entry(spec, length, script):
    c = spec["catalogue"]
    links = "\n".join(f"      {{label: {js(link['label'])}, {'to' if 'to' in link else 'href'}: {js(link.get('to') or link['href'])}}},"
                      for link in c["links"])
    paras = "\n".join(f"      {json.dumps(p, ensure_ascii=False)}," for p in transcript(script))
    return (f"  {{\n    id: {js(c['id'])},\n    title: {js(c['title'])},\n    youtube: {js(spec['youtube_id'])},\n"
            f"    duration: {js(clock(round(length)))},\n    published: {js(spec['published'])},\n"
            f"    kind: {js(c['kind'])},\n    area: {js(c['area'])},\n    voice: {js(c['voice'])},\n"
            f"    summary:\n      {js(c['summary'])},\n    links: [\n{links}\n    ],\n"
            f"    transcript: [\n{paras}\n    ],\n  }},\n")


def write_catalogue(path, entry, video_id):
    text = path.read_text(encoding="utf-8")
    start = text.find(f"  {{\n    id: {js(video_id)},")
    if start >= 0:
        end = text.index("\n  },\n", start) + len("\n  },\n")
        text = text[:start] + entry + text[end:]
        action = "replaced"
    else:
        close = text.index("];\n")
        text = text[:close] + entry + text[close:]
        action = "added"
    path.write_text(text, encoding="utf-8")
    print(f"{path}: entry {video_id} {action}")


def youtube_client():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not TOKEN.exists():
        sys.exit(f"No token at {TOKEN}. Make one with authorize.py (README.md, Publishing).")
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
        TOKEN.chmod(0o600)
    return build("youtube", "v3", credentials=creds)


def playlist_id(yt, title, create):
    request = yt.playlists().list(part="snippet", mine=True, maxResults=50)
    while request is not None:
        page = request.execute()
        for item in page.get("items", []):
            if item["snippet"]["title"] == title:
                return item["id"]
        request = yt.playlists().list_next(request, page)
    if not create:
        print(f'Playlist "{title}" not found; add the video by hand or rerun with --create-playlist.')
        return None
    made = yt.playlists().insert(part="snippet,status", body={
        "snippet": {"title": title}, "status": {"privacyStatus": "public"}}).execute()
    print(f'Created playlist "{title}"')
    return made["id"]


def upload(spec, text, files, create_playlist):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    yt = youtube_client()
    body = {
        "snippet": {"title": spec["title"], "description": text, "tags": spec.get("tags", []),
                    "categoryId": spec.get("category", "28"), "defaultLanguage": "en", "defaultAudioLanguage": "en"},
        "status": {"privacyStatus": spec.get("privacy", "private"), "selfDeclaredMadeForKids": spec.get("made_for_kids", False),
                   "license": spec.get("license", "youtube"), "containsSyntheticMedia": spec.get("synthetic_media", False),
                   "embeddable": True},
    }
    request = yt.videos().insert(part="snippet,status", body=body,
                                 media_body=MediaFileUpload(str(files["video"]), mimetype="video/mp4", chunksize=8 * 2**20, resumable=True))
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"uploaded {status.progress():.0%}")
    video_id = response["id"]
    print(f"video https://youtu.be/{video_id} ({response['status']['privacyStatus']})")
    yt.captions().insert(part="snippet", body={"snippet": {"videoId": video_id, "language": "en", "name": "English"}},
                         media_body=MediaFileUpload(str(files["subtitles"]), mimetype="text/vtt")).execute()
    print("subtitles added")
    try:
        yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(files["thumbnail"]), mimetype="image/png")).execute()
        print("thumbnail set")
    except HttpError as e:  # custom thumbnails need a verified channel
        print(f"thumbnail not set ({e.status_code}); a channel needs phone verification for custom thumbnails")
    if spec.get("playlist"):
        pid = playlist_id(yt, spec["playlist"], create_playlist)
        if pid:
            yt.playlistItems().insert(part="snippet", body={"snippet": {
                "playlistId": pid, "resourceId": {"kind": "youtube#video", "videoId": video_id}}}).execute()
            print(f'added to playlist "{spec["playlist"]}"')
    return video_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("upload_file")
    ap.add_argument("--upload", action="store_true", help="send the video to YouTube")
    ap.add_argument("--create-playlist", action="store_true", help="create the playlist when the channel has none of that name")
    ap.add_argument("--again", action="store_true", help="upload even when the upload file already holds a YouTube id")
    ap.add_argument("--catalogue", type=Path, help="write the entry into this catalogue module (videos.js)")
    args = ap.parse_args()

    path = Path(args.upload_file).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    name = path.name.removesuffix(".youtube.json")
    thumb_page = HERE / spec["thumbnail"]
    files = {"video": HERE / "out" / f"{name}.mp4", "subtitles": HERE / "out" / f"{name}.vtt",
             "step script": HERE / "videos" / f"{name}.json", "thumbnail page": thumb_page,
             "thumbnail": HERE / "out" / f"{thumb_page.stem}-thumbnail.png"}
    problems, notes = [], []
    if not files["thumbnail"].exists() and thumb_page.exists():
        subprocess.run([sys.executable, str(HERE / "thumbnail.py"), str(thumb_page)], check=True)
    length = seconds_of(files["video"]) if files["video"].exists() else 0
    marks = chapters(spec, files["subtitles"], length, problems, notes) if files["subtitles"].exists() else []
    text = description(spec, marks)
    check(spec, text, files, problems)

    out = HERE / "out" / f"{name}.youtube.txt"
    out.write_text(f"{spec['title']}\n\n{text}\n\nTags: {', '.join(spec.get('tags', []))}\n", encoding="utf-8")
    print(f"{spec['title']}  ({len(spec['title'])} characters)")
    print(f"{clock(round(length))}, {len(marks)} chapters, {len(spec.get('tags', []))} tags, "
          f"playlist {spec.get('playlist')!r}, {spec.get('privacy', 'private')}; description in {out.relative_to(HERE)}")
    for n in notes:
        print(f"note: {n}")
    for p in problems:
        print(f"problem: {p}")
    if problems:
        return 1

    if args.upload:
        if spec.get("youtube_id") and not args.again:
            sys.exit(f"Already uploaded as https://youtu.be/{spec['youtube_id']}; use --again for a new upload.")
        spec["youtube_id"] = upload(spec, text, files, args.create_playlist)
        spec["published"] = datetime.datetime.now(datetime.timezone.utc).strftime("%B %Y")  # noqa: UP017 (the video environment runs Python 3.10, which has no datetime.UTC)
        path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.relative_to(HERE)}: youtube_id {spec['youtube_id']}")
    if args.catalogue:
        if not spec.get("youtube_id"):
            sys.exit("The upload file holds no YouTube id yet; upload first.")
        write_catalogue(args.catalogue, catalogue_entry(spec, length, files["step script"]), spec["catalogue"]["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
