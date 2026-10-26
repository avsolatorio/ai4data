"""Make the OAuth token that publish.py uses to upload to a YouTube channel.

Run this once, on a computer with a browser, signed in to the Google account
that owns the channel. It needs the OAuth client file (client_secret.json) of
a Google Cloud project with the YouTube Data API v3 enabled, and the client
must be of the type "Desktop app":

    python -m pip install google-auth-oauthlib
    python authorize.py client_secret.json youtube-token.json

The browser asks for consent to manage the channel's videos. The token file
it writes lets publish.py upload without a browser; copy it to the server at
~/.config/ai4data/youtube-token.json and make it readable only by its owner
(chmod 600). It holds a refresh token: keep it out of git and out of shared
folders, and revoke it at https://myaccount.google.com/permissions when it is
no longer needed.
"""
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl"]


def main(client_file, token_file):
    flow = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES)
    creds = flow.run_local_server(port=0)
    out = Path(token_file)
    out.write_text(creds.to_json(), encoding="utf-8")
    out.chmod(0o600)
    print(f"{out} written; copy it to ~/.config/ai4data/youtube-token.json on the server")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
