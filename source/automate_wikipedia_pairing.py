# -*- coding: utf-8 -*-
import json
import wikipedia
import time
import os
import re

# --- Configuration ---
INPUT_METADATA_FILE = "./data/output/midi_metadata_list.json" # Generated in previous steps
OUTPUT_PAIRED_FILE = "./data/output/automated_paired_data.json"
REQUEST_DELAY_SECONDS = 1 # Delay between Wikipedia API calls

def clean_artist_title(text):
    """Removes underscores and potentially problematic characters for search."""
    # Replace underscores with spaces
    text = text.replace("_", " ")
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_wikipedia_summary(artist, title):
    """Searches Wikipedia for a song/artist and returns the summary and URL."""
    cleaned_artist = clean_artist_title(artist)
    cleaned_title = clean_artist_title(title)
    
    # Try specific queries first
    queries = [
        f"{cleaned_artist} {cleaned_title} (song)",
        f"{cleaned_artist} {cleaned_title}",
        f"{cleaned_title} (song)", # Sometimes title alone is enough
        f"{cleaned_artist} {cleaned_title} (album)", # Maybe it is an album title
        f"{cleaned_artist}" # Fallback to artist page
    ]
    
    summary = None
    page_url = None
    
    for query in queries:
        print(f"  Trying query: 	{query}	")
        try:
            # Search returns list of possible titles
            search_results = wikipedia.search(query)
            if not search_results:
                print(f"    No search results for query: {query}")
                continue
                
            # Try the first search result
            page_title = search_results[0]
            print(f"    Found potential page: {page_title}")
            page = wikipedia.page(page_title, auto_suggest=False) # Use exact title from search
            summary = page.summary
            page_url = page.url
            print(f"    Successfully fetched summary from: {page_url}")
            break # Stop searching if we found a summary
            
        except wikipedia.exceptions.PageError:
            print(f"    PageError: Page 	{page_title}	 not found (or search result was inaccurate).")
            continue # Try next query
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"    DisambiguationError for query 	{query}	. Options: {e.options[:5]}...")
            # Could try to fetch the first option, but might be wrong. Skipping for now.
            # try:
            #     page_title = e.options[0]
            #     page = wikipedia.page(page_title, auto_suggest=False)
            #     summary = page.summary
            #     page_url = page.url
            #     print(f"    Using first disambiguation option: {page_title}")
            #     break
            # except Exception as inner_e:
            #     print(f"    Failed to fetch first disambiguation option: {inner_e}")
            continue # Try next query
        except Exception as e:
            print(f"    An unexpected error occurred for query 	{query}	: {e}")
            # Log the error, maybe try next query
            continue
            
    if summary:
        # Basic cleaning of summary
        summary = re.sub(r"\n+", " ", summary) # Replace newlines
        summary = re.sub(r"\s+", " ", summary).strip() # Normalize whitespace
        
    return summary, page_url

# --- Main Script ---
if __name__ == "__main__":
    # Load MIDI metadata
    try:
        with open(INPUT_METADATA_FILE, "r") as f:
            midi_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input metadata file not found at {INPUT_METADATA_FILE}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_METADATA_FILE}")
        exit()

    if not midi_metadata:
        print("No metadata found in the input file.")
        exit()

    automated_paired_data = []
    print(f"Starting automated pairing for {len(midi_metadata)} MIDI files...")

    # Limit processing for demonstration if needed (e.g., first 10)
    # metadata_to_process = midi_metadata[:10]
    metadata_to_process = midi_metadata[:] 

    for i, item in enumerate(metadata_to_process):
        file_path = item.get("file_path")
        artist = item.get("artist")
        title = item.get("title")

        if not all([file_path, artist, title]):
            print(f"Skipping item {i+1} due to missing data: {item}")
            continue

        print(f"\nProcessing item {i+1}/{len(metadata_to_process)}: Artist=	{artist}	, Title=	{title}	")
        
        summary, url = get_wikipedia_summary(artist, title)
        
        paired_item = {
            "file_path": file_path,
            "artist": artist,
            "title": title,
            "wikipedia_url": url if url else "Not Found",
            "text_description": summary if summary else "Not Found"
        }
        automated_paired_data.append(paired_item)
        
        # Be polite to Wikipedia servers
        print(f"Waiting {REQUEST_DELAY_SECONDS} second(s)...")
        time.sleep(REQUEST_DELAY_SECONDS)

    # Save the results
    try:
        with open(OUTPUT_PAIRED_FILE, "w") as f:
            json.dump(automated_paired_data, f, indent=4)
        print(f"\nSuccessfully saved automated paired data to {OUTPUT_PAIRED_FILE}")
        print(f"Processed {len(automated_paired_data)} items.")
    except IOError:
        print(f"Error: Could not write results to {OUTPUT_PAIRED_FILE}")

