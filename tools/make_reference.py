#!/usr/bin/env python3
"""
Create an editable reference file for the AI Subtitle Studio.

This tool creates a CSV file with subtitle lines for human correction.
The CSV contains both ASR output and human-corrected references.
Based on Task 1.1 from docs/EXECUTOR_HANDOFF.md.
"""

import argparse
import csv
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import app modules with proper path handling
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.paths import default_data_root
from app import __version__


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_newest_refined_file(job_dir: Path) -> Optional[Path]:
    """Get the newest refined.*.json file from a job directory."""
    refined_files = list(job_dir.glob('refined.*.json'))
    if not refined_files:
        return None
    return max(refined_files, key=lambda p: p.stat().st_mtime)


def get_newest_translation_file(job_dir: Path) -> Optional[Path]:
    """Get the newest translation.*.json file from a job directory."""
    translation_files = list(job_dir.glob('translation.*.json'))
    if not translation_files:
        return None
    return max(translation_files, key=lambda p: p.stat().st_mtime)


def get_newest_master_file(job_dir: Path) -> Optional[Path]:
    """Get the newest master.*.json file from a job directory."""
    master_files = list(job_dir.glob('master.*.json'))
    if not master_files:
        return None
    return max(master_files, key=lambda p: p.stat().st_mtime)


def extract_translation_units_from_master(
    master_file: Path
) -> Dict[int, Dict[str, Any]]:
    """Extract translation units from a master JSON file.

    Args:
        master_file: Path to master JSON file

    Returns:
        Dictionary mapping unit ID to translation unit data
    """
    try:
        data = load_json_file(master_file)

        # Master transcript has translation_units as a list
        translation_units_list = data.get('translation_units')
        if isinstance(translation_units_list, list):
            translation_units = {}
            for unit in translation_units_list:
                unit_id = unit.get('id')
                if unit_id is not None:
                    translation_units[unit_id] = unit
            return translation_units

    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return {}


def extract_translation_units_from_refined(
    refined_file: Path
) -> Dict[int, Dict[str, Any]]:
    """Extract translation units from a refined JSON file.

    Args:
        refined_file: Path to refined JSON file

    Returns:
        Dictionary mapping unit ID to translation unit data
    """
    try:
        data = load_json_file(refined_file)

        # Refined files have 'units' as a list
        units_list = data.get('units')
        if isinstance(units_list, list):
            translation_units = {}
            for unit in units_list:
                unit_id = unit.get('id')
                if unit_id is not None:
                    translation_units[unit_id] = unit
            return translation_units

    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return {}


def extract_translation_units_from_translation(
    translation_file: Path
) -> Dict[int, Dict[str, Any]]:
    """Extract translation units from a translation JSON file.

    Args:
        translation_file: Path to translation JSON file

    Returns:
        Dictionary mapping unit ID to translation unit data
    """
    try:
        data = load_json_file(translation_file)

        # Translation files have 'units' as a list
        units_list = data.get('units')
        if isinstance(units_list, list):
            translation_units = {}
            for unit in units_list:
                unit_id = unit.get('id')
                if unit_id is not None:
                    translation_units[unit_id] = unit
            return translation_units

    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return {}


def extract_translation_units_from_segments(
    segments: List[Dict[str, Any]],
    master_file: Path
) -> Dict[int, Dict[str, Any]]:
    """Extract translation units from segments as fallback.

    Args:
        segments: List of segment dictionaries
        master_file: Path to master JSON file (for source language)

    Returns:
        Dictionary mapping unit ID to translation unit data
    """
    translation_units = {}

    for segment in segments:
        segment_id = segment.get('id')
        if segment_id is not None:
            translation_unit = {
                'id': segment_id,
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'source_only': True
            }
            translation_units[segment_id] = translation_unit

    return translation_units


def filter_units_by_time(
    units: Dict[int, Dict[str, Any]],
    from_seconds: float,
    to_seconds: Optional[float]
) -> Dict[int, Dict[str, Any]]:
    """Filter units by time range.

    Args:
        units: Dictionary of units by ID
        from_seconds: Start time (seconds)
        to_seconds: End time (seconds) or None for no upper limit

    Returns:
        Filtered dictionary of units
    """
    filtered_units = {}

    for unit_id, unit in units.items():
        start = unit.get('start', 0)
        end = unit.get('end', 0)

        # Check if unit overlaps with time range [from, to)
        if to_seconds is None:
            if start >= from_seconds:
                filtered_units[unit_id] = unit
        else:
            if start < to_seconds and end > from_seconds:
                filtered_units[unit_id] = unit

    return filtered_units


def create_reference_lines_from_master(
    translation_units: Dict[int, Dict[str, Any]],
    from_seconds: float,
    to_seconds: Optional[float]
) -> List[Dict[str, Any]]:
    """Create reference lines from master translation units.

    Args:
        translation_units: Dictionary of translation units
        from_seconds: Start time (seconds)
        to_seconds: End time (seconds)

    Returns:
        List of reference line dictionaries
    """
    filtered_units = filter_units_by_time(translation_units, from_seconds, to_seconds)

    reference_lines = []
    for unit_id, unit in filtered_units.items():
        line = {
            'id': unit_id,
            'start': round(unit.get('start', 0), 3),
            'end': round(unit.get('end', 0), 3),
            'source_asr': unit.get('text', ''),
            'source_ref': unit.get('text', ''),  # Pre-filled with ASR output
            'target_ai': '',  # Will be filled from final_text/translation if present
            'target_ref': '',  # Pre-filled from target_ai (will be corrected)
            'speaker_ref': '',
            'notes': ''
        }

        # Fill target_ai from final_text if available, otherwise translation
        if 'final_text' in unit:
            line['target_ai'] = unit.get('final_text', '')
            line['target_ref'] = unit.get('final_text', '')
        elif 'translation' in unit:
            line['target_ai'] = unit.get('translation', '')
            line['target_ref'] = unit.get('translation', '')

        reference_lines.append(line)

    return reference_lines


def create_reference_lines_from_refined(
    translation_units: Dict[int, Dict[str, Any]],
    from_seconds: float,
    to_seconds: Optional[float]
) -> List[Dict[str, Any]]:
    """Create reference lines from refined translation units.

    Args:
        translation_units: Dictionary of translation units
        from_seconds: Start time (seconds)
        to_seconds: End time (seconds)

    Returns:
        List of reference line dictionaries
    """
    filtered_units = filter_units_by_time(translation_units, from_seconds, to_seconds)

    reference_lines = []
    for unit_id, unit in filtered_units.items():
        line = {
            'id': unit_id,
            'start': round(unit.get('start', 0), 3),
            'end': round(unit.get('end', 0), 3),
            'source_asr': unit.get('text', ''),
            'source_ref': unit.get('text', ''),  # Pre-filled with ASR output
            'target_ai': unit.get('translation', ''),
            'target_ref': unit.get('translation', ''),  # Pre-filled from AI translation
            'speaker_ref': '',
            'notes': ''
        }

        reference_lines.append(line)

    return reference_lines


def create_reference_lines_from_translation(
    translation_units: Dict[int, Dict[str, Any]],
    from_seconds: float,
    to_seconds: Optional[float]
) -> List[Dict[str, Any]]:
    """Create reference lines from translation file units.

    Args:
        translation_units: Dictionary of translation units
        from_seconds: Start time (seconds)
        to_seconds: End time (seconds)

    Returns:
        List of reference line dictionaries
    """
    filtered_units = filter_units_by_time(translation_units, from_seconds, to_seconds)

    reference_lines = []
    for unit_id, unit in filtered_units.items():
        line = {
            'id': unit_id,
            'start': round(unit.get('start', 0), 3),
            'end': round(unit.get('end', 0), 3),
            'source_asr': unit.get('text', ''),
            'source_ref': unit.get('text', ''),  # Pre-filled with ASR output
            'target_ai': '',  # Empty target for translation file units
            'target_ref': '',  # Empty target for translation file units
            'speaker_ref': '',
            'notes': ''
        }

        reference_lines.append(line)

    return reference_lines


def write_reference_files(
    reference_lines: List[Dict[str, Any]],
    output_path: Path,
    name: str,
    job_key: Optional[str] = None,
    media_path: Optional[str] = None,
    source_language: str = 'tr',
    target_language: str = 'ar',
    app_version: str = '1.0.0',
    time_range: Optional[Dict[str, float]] = None
) -> None:
    """Write reference files to disk.

    Args:
        reference_lines: List of reference line dictionaries
        output_path: Path to output directory
        name: Reference name
        job_key: Optional job key for metadata
        media_path: Optional media path or URL
        source_language: Source language code
        target_language: Target language code
        app_version: App version
        time_range: Optional time range dictionary
    """
    # Create output directory if it does not exist (main() may have held a backup)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for atomic write
    temp_parent = output_path.parent
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
        temp_path = Path(temp_dir)

        # Write CSV file
        csv_path = temp_path / 'lines.csv'
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'id', 'start', 'end', 'source_asr', 'source_ref',
                'target_ai', 'target_ref', 'speaker_ref', 'notes'
            ])

            # Write data
            for line in reference_lines:
                writer.writerow([
                    line['id'],
                    f"{line['start']:.3f}",
                    f"{line['end']:.3f}",
                    line['source_asr'],
                    line['source_ref'],
                    line['target_ai'],
                    line['target_ref'],
                    line['speaker_ref'],
                    line['notes']
                ])

        # Write meta.json
        meta = {
            'media_path': media_path,
            'source_language': source_language,
            'target_language': target_language,
            'time_range': time_range or {'from': 0, 'to': 0},
            'job_key': job_key,
            'app_version': app_version,
            'creation_date': datetime.now().isoformat(),
            'source_file': 'lines.csv'
        }

        meta_path = temp_path / 'meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Write README.txt
        readme_path = temp_path / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""Reference File Instructions:

1. Correct source_ref to exactly what is said in the audio.
   - This should match the source audio perfectly.
   - Use the source_asr as a starting point, but correct any errors.

2. Correct target_ref to a correct, natural subtitle in the target language.
   - This should be a proper translation or appropriate subtitle text.
   - Use target_ai as a starting point, but improve grammar, fluency, and adequacy.

3. Leave a cell unchanged when it is already correct.
   - Don't modify text that is already accurate and natural.

4. Lines can be merged or split by editing the times.
   - If two consecutive lines should be one subtitle, merge them.
   - If one subtitle should be split into multiple lines, split it.

5. New rows are allowed.
   - Add additional lines if needed for completeness.

6. speaker_ref is optional: add character names here if desired.

7. Times are in seconds with 3 decimal places (e.g., 12.345).
""")

        # Move files from temp directory to final location
        for item in temp_path.iterdir():
            shutil.move(str(item), str(output_path / item.name))


def resolve_job_directory(job_arg: str) -> Path:
    """Resolve a job argument to a directory path.

    Args:
        job_arg: Job key or full path

    Returns:
        Path to job directory
    """
    # Try as full path first
    job_path = Path(job_arg)
    if job_path.is_dir():
        return job_path

    # Try as job key in default data root
    data_root = default_data_root()
    job_path = data_root / 'jobs' / job_arg
    if job_path.is_dir():
        return job_path

    # Try as job key in current working directory
    job_path = Path.cwd() / 'data' / 'jobs' / job_arg
    if job_path.is_dir():
        return job_path

    # Not found
    return job_path  # Return the path that doesn't exist


def is_reference_folder(path: Path) -> bool:
    """Check if a path looks like a reference folder (has meta.json and lines.csv)."""
    if not path.is_dir():
        return False
    return (path / 'meta.json').is_file() and (path / 'lines.csv').is_file()


def load_job_metadata(job_dir: Path) -> tuple[str, str, str]:
    """Load metadata from a job directory.

    Args:
        job_dir: Path to job directory

    Returns:
        Tuple of (source_language, target_language, media_path)
    """
    # Try to load input.json
    input_json_path = job_dir / 'input.json'
    if input_json_path.exists():
        try:
            input_data = load_json_file(input_json_path)
            source_language = input_data.get('source_language', 'tr')
            target_language = input_data.get('target_language', 'ar')
            media_path = input_data.get('input_path', '')
            return source_language, target_language, media_path
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Fallback to hardcoded values
    return 'tr', 'ar', ''


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create an editable reference file for AI Subtitle Studio (Task 1.1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create reference from a finished job folder
  python tools/make_reference.py --job 7aba9c645e5c93db --name ref1 --out reference/ --from 0 --to 60

  # Create reference from episode output folder
  python tools/make_reference.py --transcript output/<episode>/work/MasterTranscript.json --name ref2 --out reference/ --from 10

  # Overwrite existing reference
  python tools/make_reference.py --job 7aba9c645e5c93db --name ref1 --out reference/ --force
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--job',
        help='Job key (directory name in <data>/jobs/) or full path to job folder'
    )
    input_group.add_argument(
        '--transcript',
        help='Path to MasterTranscript.json file'
    )

    # Time range options
    parser.add_argument(
        '--from',
        dest='from_seconds',
        type=float,
        help='Start time for reference (seconds)'
    )

    parser.add_argument(
        '--to',
        dest='to_seconds',
        type=float,
        help='End time for reference (seconds)'
    )

    # Output options
    parser.add_argument(
        '--name',
        required=True,
        help='Reference name (CSV will be named lines.csv, not <name>.csv)'
    )

    parser.add_argument(
        '--out',
        dest='output_dir',
        default='reference',
        help='Output directory (default: reference/)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing reference folder'
    )

    args = parser.parse_args()

    try:
        # Compute target path: <out>/<name>/
        output_base = Path(args.output_dir)
        target = output_base / args.name

        # Validate target and handle force flag BEFORE processing input
        if target.exists():
            if not args.force:
                print(f"Error: Reference '{args.name}' already exists: {target}")
                print("Use --force to overwrite.")
                return 1
            # Check if it's a reference folder (has meta.json and lines.csv)
            if not is_reference_folder(target):
                print(f"Error: Target '{target}' is not a reference folder")
                print("Cannot overwrite non-reference directories.")
                return 1

        # Create output base directory if it doesn't exist
        output_base.mkdir(parents=True, exist_ok=True)

        # Determine source of input data
        if args.job:
            # Resolve job directory
            job_dir = resolve_job_directory(args.job)
            if not job_dir.exists():
                print(f"Error: Job directory not found: {args.job}")
                print("Looked in:")
                print(f"  {args.job} (full path)")
                print(f"  {default_data_root() / 'jobs' / args.job}")
                print(f"  {Path.cwd() / 'data' / 'jobs' / args.job}")
                return 1

            # Find newest refined file, then translation file, then master file
            refined_file = get_newest_refined_file(job_dir)
            translation_file = None
            master_file = None

            if refined_file:
                # Use refined file
                translation_units = extract_translation_units_from_refined(refined_file)
                source_language, target_language, media_path = load_job_metadata(job_dir)

            else:
                translation_file = get_newest_translation_file(job_dir)
                if translation_file:
                    # Use translation file
                    translation_units = extract_translation_units_from_translation(translation_file)
                    source_language, target_language, media_path = load_job_metadata(job_dir)

                else:
                    # Fallback to master file
                    master_file = get_newest_master_file(job_dir)
                    if master_file:
                        # Load master transcript
                        master_data = load_json_file(master_file)
                        segments = master_data.get('segments', [])

                        # Get source language from master
                        source_language = master_data.get('language', {}).get('code', 'tr')

                        # Get target language from job metadata
                        target_language, _, _ = load_job_metadata(job_dir)

                        # Get media path from master job data
                        job_data = master_data.get('job', {})
                        media_path = job_data.get('input_path', '')

                        # Create translation units from segments (source only)
                        translation_units = extract_translation_units_from_segments(
                            segments, master_file
                        )

                        if not translation_units:
                            print("Warning: No translation units found in job directory.")

                    else:
                        print(f"Error: No master, translation, or refined files found in {job_dir}")
                        return 1

        elif args.transcript:
            transcript_path = Path(args.transcript)
            if not transcript_path.exists():
                print(f"Error: Transcript file not found: {args.transcript}")
                return 1

            # Load master transcript
            master_data = load_json_file(transcript_path)
            segments = master_data.get('segments', [])

            # Get source language from master
            source_language = master_data.get('language', {}).get('code', 'tr')

            # Get target language from master job data
            job_data = master_data.get('job', {})
            target_language = job_data.get('target_language', 'ar')

            # Get media path from master job data
            media_path = job_data.get('input_path', transcript_path.parent / 'MasterTranscript.json')

            # Extract translation units from master
            translation_units = extract_translation_units_from_master(transcript_path)

            if not translation_units:
                print("Warning: No translation units found in master transcript.")

        # Prepare time range
        from_seconds = args.from_seconds or 0.0
        to_seconds = args.to_seconds

        # Create reference lines based on input type
        reference_lines = []

        if args.job:
            if refined_file:
                reference_lines = create_reference_lines_from_refined(
                    translation_units, from_seconds, to_seconds
                )
            elif translation_file:
                reference_lines = create_reference_lines_from_translation(
                    translation_units, from_seconds, to_seconds
                )
            elif master_file:
                reference_lines = create_reference_lines_from_master(
                    translation_units, from_seconds, to_seconds
                )
        elif args.transcript:
            reference_lines = create_reference_lines_from_master(
                translation_units, from_seconds, to_seconds
            )

        if not reference_lines:
            print("Warning: No reference lines created. Check time range.")

        # Prepare time range for metadata
        if to_seconds is None:
            # Use end of last included unit
            if reference_lines:
                to_seconds = max(line['end'] for line in reference_lines)
            else:
                to_seconds = 0.0

        time_range = {
            'from': from_seconds,
            'to': to_seconds
        }

        # Input validated; lines built. Now it's safe to touch the target.
        # If target exists (--force verified a reference folder), rename to a
        # backup, write the new one, and remove the backup on success.
        backup_name = None
        if target.exists():
            backup_name = target.parent / f"{args.name}.backup.{int(datetime.now().timestamp())}"
            target.rename(backup_name)

        try:
            write_reference_files(
            reference_lines,
            target,
            args.name,
            args.job,
            str(media_path) if media_path else None,
            source_language,
            target_language,
            __version__,
            time_range
            )
        except Exception:
            # Restore backup so the old reference is not lost
            if backup_name and backup_name.exists():
                if target.exists():
                    shutil.rmtree(target)        # the partial folder this run created
                backup_name.rename(target)
            raise

        if backup_name and backup_name.exists():
            shutil.rmtree(backup_name)

        print(f"Created reference files in: {target}")
        print(f"  CSV file: {target / 'lines.csv'}")
        print(f"  Meta file: {target / 'meta.json'}")
        print(f"  Lines: {len(reference_lines)}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
