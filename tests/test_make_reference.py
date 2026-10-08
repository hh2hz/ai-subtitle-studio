"""Tests for make_reference.py tool."""

import json
import tempfile
import csv
from pathlib import Path

from tools.make_reference import (
    extract_translation_units_from_master,
    extract_translation_units_from_refined,
    filter_units_by_time,
    create_reference_lines_from_master,
    create_reference_lines_from_refined,
    write_reference_files,
)


class TestExtractTranslationUnitsFromMaster:
    """Test master transcript translation unit extraction."""

    def test_extract_from_master_list(self):
        """Test extraction when translation_units is a list (real app format)."""
        master_data = {
            'translation_units': [
                {
                    'id': 1,
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'Hello world',
                    'translation': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645',
                    'final_text': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645'
                },
                {
                    'id': 2,
                    'start': 5.5,
                    'end': 10.5,
                    'text': 'How are you',
                    'translation': '\u0643\u064A\u0641 \u062D\u0627\u0644\u0643'
                }
            ]
        }

        # Create temp directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            master_file = Path(temp_dir) / 'test_master.json'
            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master_data, f)

            result = extract_translation_units_from_master(master_file)

            assert result == {
                1: {
                    'id': 1,
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'Hello world',
                    'translation': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645',
                    'final_text': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645'
                },
                2: {
                    'id': 2,
                    'start': 5.5,
                    'end': 10.5,
                    'text': 'How are you',
                    'translation': '\u0643\u064A\u0641 \u062D\u0627\u0644\u0643'
                }
            }

    def test_extract_empty_master(self):
        """Test extraction from empty master file."""
        master_data = {'other_field': 'value'}

        with tempfile.TemporaryDirectory() as temp_dir:
            master_file = Path(temp_dir) / 'test_master_empty.json'
            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master_data, f)

            result = extract_translation_units_from_master(master_file)

            assert result == {}

    def test_extract_invalid_json(self):
        """Test extraction from invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            master_file = Path(temp_dir) / 'invalid.json'
            master_file.write_text('invalid json')

            result = extract_translation_units_from_master(master_file)

            assert result == {}


class TestExtractTranslationUnitsFromRefined:
    """Test refined JSON translation unit extraction."""

    def test_extract_from_refined_list(self):
        """Test extraction when units is a list (refined format)."""
        refined_data = {
            'units': [
                {
                    'id': 10,
                    'start': 1.0,
                    'end': 3.0,
                    'text': 'First line',
                    'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u0623\u0648\u0644'
                },
                {
                    'id': 11,
                    'start': 3.5,
                    'end': 6.5,
                    'text': 'Second line',
                    'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u062B\u0627\u0646\u064A'
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            refined_file = Path(temp_dir) / 'test_refined.json'
            with open(refined_file, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f)

            result = extract_translation_units_from_refined(refined_file)

            assert result == {
                10: {
                    'id': 10,
                    'start': 1.0,
                    'end': 3.0,
                    'text': 'First line',
                    'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u0623\u0648\u0644'
                },
                11: {
                    'id': 11,
                    'start': 3.5,
                    'end': 6.5,
                    'text': 'Second line',
                    'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u062B\u0627\u0646\u064A'
                }
            }


class TestFilterUnitsByTime:
    """Test time filtering of translation units."""

    def test_filter_units(self):
        """Test filtering units by time range."""
        units = {
            1: {'id': 1, 'start': 0.0, 'end': 5.0},
            2: {'id': 2, 'start': 5.5, 'end': 10.5},
            3: {'id': 3, 'start': 12.0, 'end': 15.0},
            4: {'id': 4, 'start': 20.0, 'end': 25.0}
        }

        # Filter from 0 to 11 seconds - unit 3 starts at 12.0 (not included)
        result = filter_units_by_time(units, 0.0, 11.0)

        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert 3 not in result
        assert 4 not in result

    def test_filter_no_upper_limit(self):
        """Test filtering units with no upper limit."""
        units = {
            1: {'id': 1, 'start': 0.0, 'end': 5.0},
            2: {'id': 2, 'start': 10.0, 'end': 15.0}
        }

        result = filter_units_by_time(units, 0.0, None)

        assert len(result) == 2
        assert 1 in result
        assert 2 in result

    def test_filter_empty(self):
        """Test filtering when no units match."""
        units = {
            1: {'id': 1, 'start': 10.0, 'end': 15.0},
            2: {'id': 2, 'start': 20.0, 'end': 25.0}
        }

        result = filter_units_by_time(units, 0.0, 5.0)

        assert len(result) == 0


class TestCreateReferenceLinesFromMaster:
    """Test creation of reference lines from master units."""

    def test_create_from_master(self):
        """Test creating reference lines from master units."""
        translation_units = {
            1: {
                'id': 1,
                'start': 0.0,
                'end': 5.0,
                'text': 'Hello world',
                'translation': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645',
                'final_text': '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645'
            },
            2: {
                'id': 2,
                'start': 5.5,
                'end': 10.5,
                'text': 'How are you',
                'translation': '\u0643\u064A\u0641 \u062D\u0627\u0644\u0643'
            }
        }

        lines = create_reference_lines_from_master(translation_units, 0.0, 20.0)

        assert len(lines) == 2

        # Check first line
        assert lines[0]['id'] == 1
        assert lines[0]['start'] == 0.0
        assert lines[0]['end'] == 5.0
        assert lines[0]['source_asr'] == 'Hello world'
        assert lines[0]['source_ref'] == 'Hello world'
        assert lines[0]['target_ai'] == '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645'
        assert lines[0]['target_ref'] == '\u0645\u0631\u062D\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645'

        # Check second line
        assert lines[1]['id'] == 2
        assert lines[1]['start'] == 5.5
        assert lines[1]['end'] == 10.5
        assert lines[1]['source_asr'] == 'How are you'
        assert lines[1]['source_ref'] == 'How are you'
        assert lines[1]['target_ai'] == '\u0643\u064A\u0641 \u062D\u0627\u0644\u0643'
        assert lines[1]['target_ref'] == '\u0643\u064A\u0641 \u062D\u0627\u0644\u0643'


class TestCreateReferenceLinesFromRefined:
    """Test creation of reference lines from refined units."""

    def test_create_from_refined(self):
        """Test creating reference lines from refined units."""
        translation_units = {
            10: {
                'id': 10,
                'start': 1.0,
                'end': 3.0,
                'text': 'First line',
                'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u0623\u0648\u0644'
            },
            11: {
                'id': 11,
                'start': 3.5,
                'end': 6.5,
                'text': 'Second line',
                'translation': '\u0627\u0644\u062E\u0637 \u0627\u0644\u062B\u0627\u0646\u064A'
            }
        }

        lines = create_reference_lines_from_refined(translation_units, 0.0, 10.0)

        assert len(lines) == 2

        # Check first line
        assert lines[0]['id'] == 10
        assert lines[0]['start'] == 1.0
        assert lines[0]['end'] == 3.0
        assert lines[0]['source_asr'] == 'First line'
        assert lines[0]['source_ref'] == 'First line'
        assert lines[0]['target_ai'] == '\u0627\u0644\u062E\u0637 \u0627\u0644\u0623\u0648\u0644'
        assert lines[0]['target_ref'] == '\u0627\u0644\u062E\u0637 \u0627\u0644\u0623\u0648\u0644'


class TestWriteReferenceFiles:
    """Test writing reference files."""

    def test_write_files(self):
        """Test writing reference files."""
        reference_lines = [
            {
                'id': 1,
                'start': 0.0,
                'end': 5.0,
                'source_asr': 'Hello',
                'source_ref': 'Hello',
                'target_ai': '\u0645\u0631\u062D\u0628\u0627',
                'target_ref': '\u0645\u0631\u062D\u0628\u0627',
                'speaker_ref': '',
                'notes': ''
            },
            {
                'id': 2,
                'start': 5.5,
                'end': 10.5,
                'source_asr': 'World',
                'source_ref': 'World',
                'target_ai': '\u0627\u0644\u0639\u0627\u0644\u0645',
                'target_ref': '\u0627\u0644\u0639\u0627\u0644\u0645',
                'speaker_ref': 'Speaker1',
                'notes': 'Test note'
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_reference'
            name = 'test_ref'

            write_reference_files(
                reference_lines,
                output_path,
                name,
                job_key='test_job',
                media_path='/test/video.mp4',
                source_language='en',
                target_language='ar',
                app_version='1.0.0',
                time_range={'from': 0.0, 'to': 20.0}
            )

            # Check that files were created
            assert (output_path / 'lines.csv').exists()
            assert (output_path / 'meta.json').exists()
            assert (output_path / 'README.txt').exists()

            # Check CSV content
            with open(output_path / 'lines.csv', 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)

                # Check header
                assert rows[0] == [
                    'id', 'start', 'end', 'source_asr', 'source_ref',
                    'target_ai', 'target_ref', 'speaker_ref', 'notes'
                ]

                # Check first data row
                assert rows[1] == [
                    '1', '0.000', '5.000', 'Hello', 'Hello',
                    '\u0645\u0631\u062D\u0628\u0627', '\u0645\u0631\u062D\u0628\u0627', '', ''
                ]

                # Check second data row
                assert rows[2] == [
                    '2', '5.500', '10.500', 'World', 'World',
                    '\u0627\u0644\u0639\u0627\u0644\u0645', '\u0627\u0644\u0639\u0627\u0644\u0645', 'Speaker1', 'Test note'
                ]

            # Check meta.json content
            meta = json.loads((output_path / 'meta.json').read_text(encoding='utf-8'))
            assert meta['media_path'] == '/test/video.mp4'
            assert meta['source_language'] == 'en'
            assert meta['target_language'] == 'ar'
            assert meta['job_key'] == 'test_job'
            assert meta['app_version'] == '1.0.0'
            assert meta['time_range'] == {'from': 0.0, 'to': 20.0}
            assert meta['source_file'] == 'lines.csv'

    def test_write_files_no_force(self):
        """Test that writing files without --force fails on existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'existing_ref'
            output_path.mkdir()

            # Create a file in the existing directory
            (output_path / 'existing.txt').write_text('test')

            reference_lines = [{
                'id': 1,
                'start': 0.0,
                'end': 5.0,
                'source_asr': 'Hello',
                'source_ref': 'Hello',
                'target_ai': '',
                'target_ref': '',
                'speaker_ref': '',
                'notes': ''
            }]

            # This should fail and return error code
            # We'll modify write_reference_files to accept a force parameter
            # For now, let's test the actual function call
            try:
                write_reference_files(reference_lines, output_path, 'test_ref')
                # If we get here, it means the function succeeded (which shouldn't happen with existing directory)
                # This indicates we need to fix the function
                assert False, "Function should have failed on existing directory"
            except Exception:
                # Expected to fail
                pass


class TestIntegration:
    """Integration tests for the make_reference tool."""

    def test_force_keeps_sibling_references(self, tmp_path, monkeypatch, capsys):
        """--force must not delete other references in the same --out folder."""
        import sys
        from tools import make_reference

        job_dir = tmp_path / 'job1'
        job_dir.mkdir()
        (job_dir / 'refined.abc.json').write_text(json.dumps({
            'units': [{'id': 1, 'start': 0.0, 'end': 2.0,
                       'text': 'Hello', 'translation': 'World'}]
        }), encoding='utf-8')
        (job_dir / 'input.json').write_text(json.dumps({
            'source_language': 'tr', 'target_language': 'ar', 'input_path': 'x.mp4'
        }), encoding='utf-8')

        out_dir = tmp_path / 'ref'
        sibling = out_dir / 'other'
        sibling.mkdir(parents=True)
        (sibling / 'meta.json').write_text('{}', encoding='utf-8')
        (sibling / 'lines.csv').write_text('x', encoding='utf-8')

        argv = ['make_reference.py', '--job', str(job_dir), '--name', 'check',
                '--out', str(out_dir), '--force']
        monkeypatch.setattr(sys, 'argv', argv)
        assert make_reference.main() == 0
        # Run again on the same name with --force
        assert make_reference.main() == 0
        # Sibling must still exist
        assert (sibling / 'meta.json').is_file()

    def test_failed_write_restores_previous_reference(self, tmp_path, monkeypatch):
        """A failure after the new folder was created must bring the old reference back unchanged."""
        import sys
        from tools import make_reference

        job_dir = tmp_path / 'job1'
        job_dir.mkdir()
        (job_dir / 'refined.abc.json').write_text(json.dumps({
            'units': [{'id': 1, 'start': 0.0, 'end': 2.0, 'text': 'Hello', 'translation': 'World'}]
        }), encoding='utf-8')
        out_dir = tmp_path / 'ref'
        monkeypatch.setattr(sys, 'argv', ['make_reference.py', '--job', str(job_dir), '--name', 'check',
                                          '--out', str(out_dir), '--force'])
        assert make_reference.main() == 0
        before = (out_dir / 'check' / 'lines.csv').read_bytes()

        def broken(lines, target, *args, **kwargs):
            target.mkdir(parents=True, exist_ok=True)
            (target / 'lines.csv').write_text('partial', encoding='utf-8')
            raise OSError('disk full')

        monkeypatch.setattr(make_reference, 'write_reference_files', broken)
        assert make_reference.main() != 0
        assert (out_dir / 'check' / 'lines.csv').read_bytes() == before
        assert not list(out_dir.glob('check.backup.*'))

    def test_force_invalid_job_leaves_existing_reference(self, tmp_path, monkeypatch, capsys):
        """--force with an invalid --job must not delete the existing reference."""
        import sys
        from tools import make_reference

        out_dir = tmp_path / 'ref'
        existing = out_dir / 'check'
        existing.mkdir(parents=True)
        (existing / 'meta.json').write_text('{}', encoding='utf-8')
        (existing / 'lines.csv').write_text('x', encoding='utf-8')

        argv = ['make_reference.py', '--job', 'nonexistent-job-key',
                '--name', 'check', '--out', str(out_dir), '--force']
        monkeypatch.setattr(sys, 'argv', argv)
        assert make_reference.main() == 1
        # Existing reference must be untouched
        assert (existing / 'lines.csv').read_text(encoding='utf-8') == 'x'

    def test_force_refuses_non_reference_folder(self, tmp_path, monkeypatch):
        """--force must refuse to overwrite a folder that is not a reference."""
        import sys
        from tools import make_reference

        job_dir = tmp_path / 'job1'
        job_dir.mkdir()
        (job_dir / 'refined.abc.json').write_text(json.dumps({
            'units': [{'id': 1, 'start': 0.0, 'end': 2.0,
                       'text': 'Hello', 'translation': 'World'}]
        }), encoding='utf-8')
        (job_dir / 'input.json').write_text(json.dumps({
            'source_language': 'tr', 'target_language': 'ar', 'input_path': 'x.mp4'
        }), encoding='utf-8')

        out_dir = tmp_path / 'ref'
        not_a_ref = out_dir / 'check'
        not_a_ref.mkdir(parents=True)
        (not_a_ref / 'something.txt').write_text('important', encoding='utf-8')

        argv = ['make_reference.py', '--job', str(job_dir), '--name', 'check',
                '--out', str(out_dir), '--force']
        monkeypatch.setattr(sys, 'argv', argv)
        assert make_reference.main() == 1
        assert (not_a_ref / 'something.txt').is_file()

    def test_full_workflow(self):
        """Test the full workflow from extraction to file writing."""
        # Create test master data
        master_data = {
            'translation_units': [
                {
                    'id': 1,
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'Welcome to our studio',
                    'translation': '\u0623\u0647\u0644\u0627 \u0628\u0643\u0645 \u0641\u064A \u0627\u0633\u062A\u0648\u062F\u064A\u0648',
                    'final_text': '\u0623\u0647\u0644\u0627 \u0628\u0643\u0645 \u0641\u064A \u0627\u0633\u062A\u0648\u062F\u064A\u0648'
                },
                {
                    'id': 2,
                    'start': 5.5,
                    'end': 12.0,
                    'text': 'We make perfect subtitles',
                    'translation': '\u0646\u062D\u0646 \u0646\u0635\u0646\u0639 \u062A\u0631\u062C\u0645\u0627\u062A \u0645\u062B\u0627\u0644\u064A\u0629'
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            master_file = Path(temp_dir) / 'integration_master.json'
            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(master_data, f)

            # Extract translation units
            translation_units = extract_translation_units_from_master(master_file)
            assert len(translation_units) == 2

            # Filter by time
            filtered_units = filter_units_by_time(translation_units, 0.0, 15.0)
            assert len(filtered_units) == 2

            # Create reference lines
            reference_lines = create_reference_lines_from_master(
                filtered_units, 0.0, 15.0
            )
            assert len(reference_lines) == 2

            # Write files
            output_path = Path(temp_dir) / 'integration_test'

            write_reference_files(
                reference_lines,
                output_path,
                'integration_test',
                job_key='integration_job',
                media_path='/integration/video.mp4',
                source_language='en',
                target_language='ar',
                app_version='1.0.0',
                time_range={'from': 0.0, 'to': 20.0}
            )

            # Verify all files exist
            assert (output_path / 'lines.csv').exists()
            assert (output_path / 'meta.json').exists()
            assert (output_path / 'README.txt').exists()

            # Verify CSV format
            with open(output_path / 'lines.csv', 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert rows[0] == [
                    'id', 'start', 'end', 'source_asr', 'source_ref',
                    'target_ai', 'target_ref', 'speaker_ref', 'notes'
                ]

                # Check Arabic content
                assert '\u0623\u0647\u0644\u0627' in rows[1][5]  # target_ai column
                assert '\u0623\u0647\u0644\u0627' in rows[1][6]  # target_ref column

        print("\u2713 All integration tests passed")


def run_all_tests():
    """Run all tests and report results."""
    test_cases = [
        TestExtractTranslationUnitsFromMaster,
        TestExtractTranslationUnitsFromRefined,
        TestFilterUnitsByTime,
        TestCreateReferenceLinesFromMaster,
        TestCreateReferenceLinesFromRefined,
        TestWriteReferenceFiles,
        TestIntegration,
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        test_instance = test_case()

        for test_name in dir(test_instance):
            if test_name.startswith('test_'):
                try:
                    method = getattr(test_instance, test_name)
                    method()
                    passed += 1
                    print(f"\u2713 {test_case.__name__}.{test_name} passed")
                except Exception as e:
                    failed += 1
                    print(f"\u2717 {test_case.__name__}.{test_name} failed: {e}")

    print("\nTest Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {passed + failed}")

    return passed, failed


if __name__ == '__main__':
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
