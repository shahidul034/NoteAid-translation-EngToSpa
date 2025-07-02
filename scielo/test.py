import xml.etree.ElementTree as ET
import json
import re
import argparse

def clean_text(text):
    """
    Collapse multiple whitespace characters into a single space and trim.
    This ensures sentences are joined cleanly without disturbing biomedical terms.
    """
    return re.sub(r'\s+', ' ', text).strip()

def parse_bioc(file_path, desired_lang, valid_sections={"title", "abstract"}):
    """
    Parse a BioC XML file to extract document-level text for a given language.
    
    Args:
        file_path (str): Path to the BioC XML file.
        desired_lang (str): The language code to extract (e.g., "EN" or "ES").
        valid_sections (set): Sections to include (by default: title and abstract).
    
    Returns:
        dict: Mapping from document id to the concatenated, cleaned text.
    """
    docs = {}
    # Use iterparse to read the XML file incrementally.
    for event, elem in ET.iterparse(file_path, events=("end",)):
        if elem.tag == "document":
            doc_id = elem.findtext("id")
            text_parts = []
            for passage in elem.findall("passage"):
                # Extract language and section from <infon> tags.
                lang = ""
                section = ""
                for infon in passage.findall("infon"):
                    key = infon.attrib.get("key", "").strip()
                    if key == "language" and infon.text:
                        lang = infon.text.strip()
                    if key == "section" and infon.text:
                        section = infon.text.strip()
                if lang == desired_lang and section in valid_sections:
                    sentences = [clean_text(s.findtext("text")) 
                                 for s in passage.findall("sentence") 
                                 if s.findtext("text")]
                    if sentences:
                        text_parts.append(" ".join(sentences))
            full_text = clean_text(" ".join(text_parts))
            if doc_id and full_text:
                docs[doc_id] = full_text
            elem.clear()  # free memory for parsed element
    return docs

def parse_bioc_grouped(file_path, valid_sections={"title", "abstract"}):
    """
    Parse a single BioC XML file that contains bilingual document pairs.
    Documents occur as separate <document> elements with the same <id>
    but different languages.
    
    Args:
        file_path (str): Path to the bilingual XML file.
        valid_sections (set): Sections to include (default: title and abstract).
    
    Returns:
        tuple: (grouped_docs, languages_found)
            - grouped_docs: Mapping from document id to a dict with language texts
            - languages_found: Set of language codes found in the file
    """
    grouped = {}
    # Track all languages encountered
    languages_found = set()
    
    for event, elem in ET.iterparse(file_path, events=("end",)):
        if elem.tag == "document":
            doc_id = elem.findtext("id")
            if not doc_id:
                continue
            lang_text_parts = {}
            # Loop over all passages in the document.
            for passage in elem.findall("passage"):
                passage_lang = ""
                passage_section = ""
                for infon in passage.findall("infon"):
                    key = infon.attrib.get("key", "").strip()
                    if key == "language" and infon.text:
                        passage_lang = infon.text.strip()
                        languages_found.add(passage_lang)
                    if key == "section" and infon.text:
                        passage_section = infon.text.strip()
                if passage_lang and passage_section in valid_sections:
                    sentences = [clean_text(s.findtext("text")) 
                                 for s in passage.findall("sentence") 
                                 if s.findtext("text")]
                    if sentences:
                        # Append or create the text list for this language.
                        lang_text_parts.setdefault(passage_lang, []).append(" ".join(sentences))
            # Join collected passages for each language.
            for lang in lang_text_parts:
                lang_text_parts[lang] = clean_text(" ".join(lang_text_parts[lang]))
            # Group by document id.
            if doc_id in grouped:
                grouped[doc_id].update(lang_text_parts)
            else:
                grouped[doc_id] = lang_text_parts
            elem.clear()
    return grouped, languages_found

def process_training_data_single_file(input_file, output_file, pivot_lang="EN"):
    """
    Process training data from a single XML file containing bilingual document pairs.
    Outputs document-level pairs from pivot language to all other languages as JSON lines.
    
    Args:
        input_file (str): Path to the input XML file.
        output_file (str): Path to the output JSON Lines file.
        pivot_lang (str): The pivot language to create pairs from (default: "EN").
    """
    grouped, languages_found = parse_bioc_grouped(input_file, valid_sections={"title", "abstract"})
    
    # Check if pivot language exists in data
    if languages_found and pivot_lang not in languages_found:
        print(f"Warning: Pivot language '{pivot_lang}' not found in data. Available languages: {', '.join(sorted(languages_found))}")
        print(f"Defaulting to first language found: '{next(iter(sorted(languages_found)))}'")
        pivot_lang = next(iter(sorted(languages_found)))
    
    pairs_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for doc_id, texts in grouped.items():
            # Check if the pivot language exists for this document
            if pivot_lang in texts:
                pivot_text = texts[pivot_lang]
                # Create pairs between pivot language and all other languages
                for lang, text in texts.items():
                    if lang != pivot_lang:  # Skip self-pairing
                        entry = {
                            "id": doc_id,
                            pivot_lang.lower(): pivot_text,
                            lang.lower(): text
                        }
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        pairs_count += 1
    
    print(f"Processed training data: {pairs_count} document pairs written to '{output_file}'.")
    print(f"Languages found in data: {', '.join(sorted(languages_found))}")

def process_training_data_multiple_files(pivot_file, other_files, output_file, pivot_lang="EN"):
    """
    Process training data from multiple XML files (one per language).
    Aligns documents by their ID and outputs pivot language to other language pairs.
    
    Args:
        pivot_file (str): Path to the pivot language XML file.
        other_files (dict): Dictionary mapping language codes to file paths.
        output_file (str): Path to the output JSON Lines file.
        pivot_lang (str): The pivot language code (default: "EN").
    """
    # Parse pivot language file
    pivot_docs = parse_bioc(pivot_file, pivot_lang, valid_sections={"title", "abstract"})
    print(f"Found {len(pivot_docs)} documents in the pivot language ({pivot_lang}) file.")
    
    # Process each target language file
    pairs_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for lang, file_path in other_files.items():
            target_docs = parse_bioc(file_path, lang, valid_sections={"title", "abstract"})
            common_ids = set(pivot_docs.keys()) & set(target_docs.keys())
            lang_pairs = len(common_ids)
            print(f"Found {lang_pairs} common document IDs between {pivot_lang} and {lang} files.")
            
            # Create pairs for this language
            for doc_id in sorted(common_ids):
                entry = {
                    "id": doc_id,
                    pivot_lang.lower(): pivot_docs[doc_id],
                    lang.lower(): target_docs[doc_id]
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                pairs_count += 1
    
    print(f"Processed training data: {pairs_count} document pairs written to '{output_file}'.")

def process_data(mode, pivot_file, other_files, output_file, pivot_lang="EN"):
    """
    Unified function to process data in both train and test modes with multiple files.
    
    Args:
        mode (str): "train" or "test" mode.
        pivot_file (str): Path to the pivot language XML file.
        other_files (dict): Dictionary mapping language codes to file paths.
        output_file (str): Path to the output JSON Lines file.
        pivot_lang (str): The pivot language code (default: "EN").
    """
    # Parse pivot language file
    pivot_docs = parse_bioc(pivot_file, pivot_lang, valid_sections={"title", "abstract"})
    print(f"Found {len(pivot_docs)} documents in the pivot language ({pivot_lang}) file.")
    
    # Process each target language file
    pairs_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for lang, file_path in other_files.items():
            target_docs = parse_bioc(file_path, lang, valid_sections={"title", "abstract"})
            common_ids = set(pivot_docs.keys()) & set(target_docs.keys())
            lang_pairs = len(common_ids)
            print(f"Found {lang_pairs} common document IDs between {pivot_lang} and {lang} files.")
            
            # Create pairs for this language
            for doc_id in sorted(common_ids):
                entry = {
                    "id": doc_id,
                    pivot_lang.lower(): pivot_docs[doc_id],
                    lang.lower(): target_docs[doc_id]
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                pairs_count += 1
    
    print(f"Processed {mode} data: {pairs_count} document pairs written to '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Process BioC XML files to produce document-level bilingual JSON pairs."
    )
    parser.add_argument(
        "--mode", choices=["train", "test", "single"], required=True,
        help=("Mode: 'train' or 'test' for multiple files by language, "
              "'single' for a single XML file with bilingual documents.")
    )
    parser.add_argument(
        "--input", required=True,
        help=("For 'single' mode: path to the bilingual XML file. "
              "For 'train'/'test' mode: path to the pivot language XML file.")
    )
    parser.add_argument(
        "--pivot_lang", default="EN",
        help="Pivot language for creating pairs (default: EN)."
    )
    parser.add_argument(
        "--lang_inputs", nargs="+", 
        help=("For 'train'/'test' mode: additional language inputs in format LANG:PATH. "
              "Example: --lang_inputs ES:/path/to/spanish.xml FR:/path/to/french.xml")
    )
    parser.add_argument(
        "--output", required=True, help="Path to output JSON Lines file."
    )
    args = parser.parse_args()

    if args.mode == "single":
        # Process a single file containing multiple languages
        process_training_data_single_file(args.input, args.output, pivot_lang=args.pivot_lang)
    else:  # train or test mode with multiple files
        if not args.lang_inputs:
            parser.error(f"In '{args.mode}' mode with multiple files, --lang_inputs must be provided.")
        
        # Parse language inputs
        other_files = {}
        for lang_input in args.lang_inputs:
            try:
                lang, path = lang_input.split(":", 1)
                other_files[lang] = path
            except ValueError:
                parser.error(f"Invalid language input format: {lang_input}. Use LANG:PATH format.")
        
        # Unified processing for train and test modes with multiple files
        process_data(args.mode, args.input, other_files, args.output, pivot_lang=args.pivot_lang)

if __name__ == "__main__":
    main()