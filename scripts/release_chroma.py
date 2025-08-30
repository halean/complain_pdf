#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import subprocess


def make_tar_gz(src_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(str(src_dir), arcname=src_dir.name)


def detect_repo() -> Optional[str]:
    # Prefer env (GitHub Actions) then git remote
    repo = os.getenv("GITHUB_REPOSITORY")
    if repo:
        return repo
    try:
        url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        # Normalize git@github.com:owner/repo.git or https://github.com/owner/repo.git
        if url.endswith(".git"):
            url = url[:-4]
        if url.startswith("git@github.com:"):
            return url.split(":", 1)[1]
        if "github.com/" in url:
            return url.split("github.com/", 1)[1]
    except Exception:
        return None
    return None


def api_request(method: str, url: str, token: str, data: Optional[bytes] = None, headers: Optional[dict] = None):
    headers = headers or {}
    headers.setdefault("Accept", "application/vnd.github+json")
    headers.setdefault("Authorization", f"Bearer {token}")
    headers.setdefault("User-Agent", "chroma-release-script/1.0")
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req) as resp:
        content = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        if ctype.startswith("application/json"):
            return json.loads(content.decode())
        return content


def create_release(repo: str, token: str, tag: str, name: Optional[str], draft: bool, body: Optional[str]):
    url = f"https://api.github.com/repos/{repo}/releases"
    payload = {
        "tag_name": tag,
        "name": name or tag,
        "draft": draft,
        "body": body or "",
    }
    return api_request("POST", url, token, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


def upload_asset(upload_url_template: str, token: str, asset_path: Path):
    base = upload_url_template.split("{", 1)[0]
    url = f"{base}?name={quote(asset_path.name)}"
    data = asset_path.read_bytes()
    headers = {
        "Content-Type": "application/gzip",
        "Content-Length": str(len(data)),
    }
    return api_request("POST", url, token, data=data, headers=headers)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Package chroma dir and upload to a GitHub Release")
    p.add_argument("--dir", default="chromadb_all_f", help="Directory to archive (default: chromadb_all_f)")
    p.add_argument("--out", default=None, help="Output .tar.gz path (default: dist/<dir>-<timestamp>.tar.gz)")
    p.add_argument("--repo", default=None, help="GitHub repo in owner/repo format (auto-detect if omitted)")
    p.add_argument("--tag", default=None, help="Release tag (default: v<YYYYmmdd-HHMMSS>)")
    p.add_argument("--name", default=None, help="Release name (default: tag)")
    p.add_argument("--body", default=None, help="Release notes body text")
    p.add_argument("--draft", action="store_true", help="Create release as draft")
    p.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub token (or env GITHUB_TOKEN)")
    args = p.parse_args(argv)

    src = Path(args.dir).resolve()
    if not src.exists() or not src.is_dir():
        print(f"Error: directory not found: {src}", file=sys.stderr)
        return 2

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("dist") / f"{src.name}-{ts}.tar.gz"
    print(f"Packaging {src} -> {out}")
    make_tar_gz(src, out)

    repo = args.repo or detect_repo()
    if not repo:
        print("Error: Could not determine repo. Pass --repo owner/repo or set GITHUB_REPOSITORY.", file=sys.stderr)
        return 2
    tag = args.tag or f"v{ts}"
    if not args.token:
        print("Error: Provide --token or set GITHUB_TOKEN.", file=sys.stderr)
        return 2

    print(f"Creating release {tag} on {repo}")
    try:
        release = create_release(repo, args.token, tag, args.name, args.draft, args.body)
    except HTTPError as e:
        try:
            print(f"HTTPError creating release: {e.code} {e.reason} {e.read().decode()}")
        except Exception:
            print(f"HTTPError creating release: {e}")
        return 1
    except URLError as e:
        print(f"URLError creating release: {e}")
        return 1

    upload_url = release.get("upload_url", "")
    if not upload_url:
        print("Error: No upload_url in release response.")
        return 1

    print(f"Uploading asset {out.name}")
    try:
        asset = upload_asset(upload_url, args.token, out)
    except HTTPError as e:
        try:
            print(f"HTTPError uploading asset: {e.code} {e.reason} {e.read().decode()}")
        except Exception:
            print(f"HTTPError uploading asset: {e}")
        return 1
    except URLError as e:
        print(f"URLError uploading asset: {e}")
        return 1

    browser_url = release.get("html_url")
    asset_url = asset.get("browser_download_url") if isinstance(asset, dict) else None
    print("Done.")
    if browser_url:
        print(f"Release page: {browser_url}")
    if asset_url:
        print(f"Asset URL: {asset_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

