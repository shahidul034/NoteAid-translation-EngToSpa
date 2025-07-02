import os
import json
from pathlib import Path
import re

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Successfully read {len(lines)} lines from {file_path}")
            return lines
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []

def format_parallel_data(src_file, tgt_file, src_lang, tgt_lang, output_file):
    print(f"\nProcessing files:")
    print(f"Source file: {src_file}")
    print(f"Target file: {tgt_file}")
    print(f"Output file: {output_file}")
    
    # Read both language files
    src_lines = read_file(src_file)
    tgt_lines = read_file(tgt_file)
    
    if not src_lines or not tgt_lines:
        print("Skipping due to empty input files")
        return
    
    # Create formatted data
    formatted_data = []
    for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        entry = {
            'id': str(i + 1),
            src_lang: src.strip(),
            tgt_lang: tgt.strip()
        }
        formatted_data.append(entry)
    
    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully wrote {len(formatted_data)} entries to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {str(e)}")

def find_parallel_files():
    parallel_pairs = []
    
    # Get all files from testsets directory
    testset_files = {}
    for f in os.listdir('testsets'):
        if f.endswith('.txt'):
            # Extract language codes from filename (e.g., de2en_de.txt -> de)
            match = re.match(r'(.+)_([a-z]{2})\.txt$', f)
            if match:
                base, src_lang = match.groups()
                testset_files[base] = (f, src_lang)
    
    # Match with goldset files
    for f in os.listdir('goldsets'):
        if f.endswith('.txt'):
            match = re.match(r'(.+)_([a-z]{2})\.txt$', f)
            if match:
                base, tgt_lang = match.groups()
                if base in testset_files:
                    src_file, src_lang = testset_files[base]
                    parallel_pairs.append({
                        'base': base,
                        'src_file': os.path.join('testsets', src_file),
                        'tgt_file': os.path.join('goldsets', f),
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang
                    })
    
    return parallel_pairs

def main():
    output_dir = Path('formatted_data')
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    parallel_pairs = find_parallel_files()
    
    for pair in parallel_pairs:
        output_file = output_dir / f"{pair['base']}_{pair['src_lang']}_{pair['tgt_lang']}.json"
        format_parallel_data(
            pair['src_file'],
            pair['tgt_file'],
            pair['src_lang'],
            pair['tgt_lang'],
            output_file
        )

if __name__ == '__main__':
    main() 