from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bioxrep.data.public_sources import PUBLIC_SOURCES, PublicSource


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def source_to_dict(source: PublicSource) -> Dict[str, object]:
    return {
        "key": source.key,
        "url": source.url,
        "filename": source.filename,
        "description": source.description,
        "track": source.track,
        "compressed": source.compressed,
        "credentialed": source.credentialed,
    }


def render_sources(sources: Iterable[PublicSource]) -> str:
    rows = [source_to_dict(source) for source in sources]
    return json.dumps(rows, indent=2, sort_keys=True)


def credential_header(source: PublicSource) -> Dict[str, str]:
    if not source.credentialed:
        return {}

    username = os.environ.get("PHYSIONET_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            f"{source.key} requires credentialed PhysioNet access. "
            "Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD after your PhysioNet account "
            "has approved MIMIC-IV access."
        )

    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def download_url(
    source: PublicSource,
    output_path: Path,
    overwrite: bool = False,
) -> Dict[str, object]:
    if output_path.exists() and not overwrite:
        return {
            "path": str(output_path),
            "status": "exists",
            "bytes": output_path.stat().st_size,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "BioXRep/0.1"}
    headers.update(credential_header(source))
    request = Request(source.url, headers=headers)

    try:
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent)) as tmp:
                shutil.copyfileobj(response, tmp)
                tmp_path = Path(tmp.name)
    except HTTPError as exc:
        if source.credentialed and exc.code in {401, 403}:
            raise RuntimeError(
                f"Failed to download {source.key}: HTTP {exc.code}. "
                "For credentialed PhysioNet sources, confirm that PHYSIONET_USERNAME "
                "and PHYSIONET_PASSWORD are correct, your PhysioNet account has approved "
                "access to the dataset, and the MIMIC-IV data use agreement is complete."
            ) from exc
        raise RuntimeError(f"Failed to download {source.url}: {exc}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download {source.url}: {exc}") from exc

    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "status": "downloaded",
        "bytes": output_path.stat().st_size,
    }


def write_fetch_manifest(records: List[Dict[str, object]], output_dir: Path) -> Path:
    manifest_path = output_dir / "fetch_manifest.json"
    existing_records: Dict[str, Dict[str, object]] = {}
    if manifest_path.exists():
        existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in existing_payload.get("records", []):
            key = str(record.get("key", ""))
            if key:
                existing_records[key] = record

    for record in records:
        existing_records[str(record["key"])] = record

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": list(existing_records.values()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def selected_sources(keys: List[str]) -> List[PublicSource]:
    if keys == ["all"]:
        return list(PUBLIC_SOURCES.values())

    missing = [key for key in keys if key not in PUBLIC_SOURCES]
    if missing:
        available = ", ".join(sorted(PUBLIC_SOURCES))
        raise ValueError(f"Unknown source(s): {', '.join(missing)}. Available: {available}")

    return [PUBLIC_SOURCES[key] for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public BioXRep source datasets.")
    parser.add_argument(
        "sources",
        nargs="*",
        default=["clinvar_variant_summary", "clinvar_hgvs", "clinvar_allele_gene", "hgnc_complete_set"],
        help="Source keys to fetch, or 'all'. Use --list to inspect available sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--list", action="store_true", help="List known public sources and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned downloads without fetching files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--env-file", type=Path, default=None, help="Load credentials from a local KEY=value file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.env_file is not None:
        load_env_file(args.env_file)

    if args.list:
        print(render_sources(PUBLIC_SOURCES.values()))
        return

    sources = selected_sources(args.sources)
    planned = []
    for source in sources:
        output_path = args.output_dir / source.key / source.filename
        planned.append({**source_to_dict(source), "output_path": str(output_path)})

    if args.dry_run:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return

    records: List[Dict[str, object]] = []
    for source in sources:
        output_path = args.output_dir / source.key / source.filename
        result = download_url(source, output_path, overwrite=args.overwrite)
        record = {**source_to_dict(source), **result}
        records.append(record)
        print(f"{record['status']}: {source.key} -> {record['path']} ({record['bytes']} bytes)")

    manifest_path = write_fetch_manifest(records, args.output_dir)
    print(f"Wrote fetch manifest to {manifest_path}")


if __name__ == "__main__":
    main()
