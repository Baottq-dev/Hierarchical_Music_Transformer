import json
import wikipedia
import time
import os
import re

def clean_artist_title(text):
    """Removes underscores and potentially problematic characters for search."""
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_wikipedia_summary(artist, title):
    """
    Searches Wikipedia for a song/artist and returns the summary and URL.
    Returns (summary, page_url) tuple.
    """
    cleaned_artist = clean_artist_title(artist)
    cleaned_title = clean_artist_title(title)
    
    queries = [
        f"{cleaned_artist} {cleaned_title} (song)",
        f"{cleaned_artist} {cleaned_title}",
        f"{cleaned_title} (song)",
        f"{cleaned_artist} {cleaned_title} (album)",
        f"{cleaned_artist}"
    ]
    
    summary = None
    page_url = None
    
    for query in queries:
        print(f"  Trying query: {query}")
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                print(f"    No search results for query: {query}")
                continue
                
            page_title = search_results[0]
            print(f"    Found potential page: {page_title}")
            page = wikipedia.page(page_title, auto_suggest=False)
            summary = page.summary
            page_url = page.url
            print(f"    Successfully fetched summary from: {page_url}")
            break
            
        except wikipedia.exceptions.PageError:
            print(f"    PageError: Page {page_title} not found.")
            continue
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"    DisambiguationError for query {query}. Options: {e.options[:5]}...")
            continue
        except Exception as e:
            print(f"    An unexpected error occurred for query {query}: {e}")
            continue
            
    if summary:
        summary = re.sub(r"\n+", " ", summary)
        summary = re.sub(r"\s+", " ", summary).strip()
        
    return summary, page_url

def pair_midi_with_wikipedia(metadata_file, output_file, request_delay=1):
    """
    Pairs MIDI metadata with Wikipedia descriptions.
    Args:
        metadata_file: Path to JSON file containing MIDI metadata
        output_file: Path to save paired data
        request_delay: Delay between Wikipedia API calls in seconds
    """
    try:
        with open(metadata_file, "r") as f:
            midi_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input metadata file not found at {metadata_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_file}")
        return

    if not midi_metadata:
        print("No metadata found in the input file.")
        return

    automated_paired_data = []
    print(f"Starting automated pairing for {len(midi_metadata)} MIDI files...")

    for i, item in enumerate(midi_metadata):
        file_path = item.get("file_path")
        artist = item.get("artist")
        title = item.get("title")

        if not all([file_path, artist, title]):
            print(f"Skipping item {i+1} due to missing data: {item}")
            continue

        print(f"\nProcessing item {i+1}/{len(midi_metadata)}: Artist={artist}, Title={title}")
        
        summary, url = get_wikipedia_summary(artist, title)
        
        paired_item = {
            "file_path": file_path,
            "artist": artist,
            "title": title,
            "wikipedia_url": url if url else "Not Found",
            "text_description": summary if summary else "Not Found"
        }
        automated_paired_data.append(paired_item)
        
        print(f"Waiting {request_delay} second(s)...")
        time.sleep(request_delay)

    try:
        with open(output_file, "w") as f:
            json.dump(automated_paired_data, f, indent=4)
        print(f"\nSuccessfully saved automated paired data to {output_file}")
        print(f"Processed {len(automated_paired_data)} items.")
    except IOError:
        print(f"Error: Could not write results to {output_file}") 