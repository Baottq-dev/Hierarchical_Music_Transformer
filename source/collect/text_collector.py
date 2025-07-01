"""
Text Collector - Collects text descriptions for MIDI files
"""

import json
import requests
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re

class TextCollector:
    """Collects text descriptions for MIDI files from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia for information about a song/artist."""
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'utf8': 1
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['query']['search']:
                page_id = data['query']['search'][0]['pageid']
                return self._get_wikipedia_content(page_id)
            
            return None
        except Exception as e:
            print(f"Error searching Wikipedia for '{query}': {e}")
            return None
    
    def _get_wikipedia_content(self, page_id: int) -> Optional[str]:
        """Get content from Wikipedia page."""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'exintro': 1,
                'explaintext': 1,
                'pageids': page_id
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'extract' in data['query']['pages'][str(page_id)]:
                return data['query']['pages'][str(page_id)]['extract']
            
            return None
        except Exception as e:
            print(f"Error getting Wikipedia content: {e}")
            return None
    
    def generate_description_from_filename(self, filename: str) -> str:
        """Generate a description from MIDI filename."""
        # Remove file extension
        name = filename.replace('.mid', '').replace('.midi', '')
        
        # Split by common separators
        parts = re.split(r'[-_\s]+', name)
        
        # Try to identify artist and song
        if len(parts) >= 2:
            artist = parts[0]
            song = ' '.join(parts[1:])
            return f"A song by {artist} titled '{song}'. This is a musical composition in MIDI format."
        else:
            return f"A musical composition titled '{name}'. This is a MIDI file containing musical notes and timing information."
    
    def collect_text_for_midi(self, midi_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect text description for a MIDI file."""
        filename = midi_metadata['file_name']
        name_without_ext = filename.replace('.mid', '').replace('.midi', '')
        
        # Try Wikipedia first
        wiki_content = self.search_wikipedia(name_without_ext)
        
        if wiki_content:
            # Clean and truncate Wikipedia content
            description = self._clean_text(wiki_content)
            source = "wikipedia"
        else:
            # Generate from filename
            description = self.generate_description_from_filename(filename)
            source = "generated"
        
        return {
            'midi_file': midi_metadata['file_path'],
            'text_description': description,
            'source': source,
            'metadata': midi_metadata
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and truncate text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text
    
    def collect_text_for_all_midi(self, midi_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect text descriptions for all MIDI files."""
        paired_data = []
        
        for metadata in midi_metadata_list:
            paired_item = self.collect_text_for_midi(metadata)
            paired_data.append(paired_item)
            
            # Add delay to be respectful to APIs
            time.sleep(0.5)
        
        return paired_data

def pair_midi_with_wikipedia(metadata_file: str, output_file: str):
    """Convenience function to pair MIDI files with Wikipedia descriptions."""
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        midi_metadata = json.load(f)
    
    # Collect text descriptions
    collector = TextCollector()
    paired_data = collector.collect_text_for_all_midi(midi_metadata)
    
    # Save paired data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)
    
    print(f"Paired data saved to {output_file}")
    print(f"Processed {len(paired_data)} MIDI files") 