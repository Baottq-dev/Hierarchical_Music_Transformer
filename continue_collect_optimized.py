#!/usr/bin/env python3
"""
Script tiếp tục thu thập dữ liệu từ metadata MIDI với tối ưu cho Colab
"""

import os
import json
import time
import argparse
import copy
from pathlib import Path
import requests

from amt.collect import DataPairing
from amt.collect.text_collector import TextCollector
from amt.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


class OptimizedTextCollector(TextCollector):
    """
    TextCollector tối ưu hóa với chức năng checkpoint và zero delay
    """
    def __init__(self, checkpoint_interval=10, delay=0.0):
        super().__init__()
        self.checkpoint_interval = checkpoint_interval
        self.delay = delay
        self.checkpoint_path = None
        self.checkpoint_filename = "paired_data_checkpoint.json"
        
    def collect_text_for_all_midi(self, midi_metadata_list, checkpoint_dir=None):
        """Collect text descriptions with checkpoint and optimized delay"""
        paired_data = []
        total_files = len(midi_metadata_list)
        
        # Set up checkpoint directory
        if checkpoint_dir:
            self.checkpoint_path = os.path.join(checkpoint_dir, self.checkpoint_filename)
            
            # Try to load from checkpoint if exists
            if os.path.exists(self.checkpoint_path):
                try:
                    with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                        paired_data = json.load(f)
                    
                    # Find where we left off
                    processed_files = set(item["midi_file"] for item in paired_data)
                    remaining_metadata = [m for m in midi_metadata_list 
                                        if m["file_path"] not in processed_files]
                    
                    if remaining_metadata:
                        logger.info(f"📋 Resuming from checkpoint: {len(paired_data)}/{total_files} files already processed")
                        midi_metadata_list = remaining_metadata
                    else:
                        logger.info(f"✅ All {total_files} files already processed according to checkpoint")
                        return paired_data
                except Exception as e:
                    logger.warning(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")
                    paired_data = []
        
        start_time = time.time()
        for i, metadata in enumerate(midi_metadata_list):
            try:
                # Show progress
                current_total = len(paired_data) + i + 1
                elapsed = time.time() - start_time
                items_per_sec = current_total / elapsed if elapsed > 0 else 0
                remaining = (total_files - current_total) / items_per_sec if items_per_sec > 0 else "unknown"
                if isinstance(remaining, float):
                    remaining_str = f"{remaining:.1f} seconds"
                else:
                    remaining_str = str(remaining)
                
                logger.info(f"Processing file {current_total}/{total_files} ({items_per_sec:.2f} files/sec, est. remaining: {remaining_str})")
                
                # Collect text
                paired_item = self.collect_text_for_midi(metadata)
                paired_data.append(paired_item)
                
                # Save checkpoint if needed
                if self.checkpoint_path and (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(paired_data)
                    logger.info(f"💾 Checkpoint saved at {i+1}/{len(midi_metadata_list)} files")
                
                # No delay between requests
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error processing file {metadata.get('file_path', 'unknown')}: {e}")
                # Continue with next file
        
        # Final checkpoint save
        if self.checkpoint_path:
            self._save_checkpoint(paired_data)
            logger.info(f"💾 Final checkpoint saved with {len(paired_data)} items")
        
        return paired_data
    
    def _save_checkpoint(self, paired_data):
        """Save checkpoint safely"""
        if not self.checkpoint_path:
            return
            
        # Create temp file
        temp_path = self.checkpoint_path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(paired_data, f, indent=2, ensure_ascii=False)
                
            # Rename temp file to actual checkpoint file
            os.replace(temp_path, self.checkpoint_path)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")


def continue_from_metadata(
    metadata_file, 
    output_dir,
    checkpoint_interval=10,
    delay=0.0,
    filter_quality=True,
    min_text_length=20,
    min_duration=10.0,
    log_level="INFO"
):
    """Continue collecting text and pairing from an existing metadata file"""
    # Set log level
    logger.setLevel(log_level)
    
    logger.info("🔄 Tiếp tục thu thập dữ liệu từ metadata...")
    start_time = time.time()
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            midi_metadata = json.load(f)
    except Exception as e:
        logger.error(f"❌ Không thể đọc file metadata: {e}")
        return None
    
    logger.info(f"✅ Đã tải metadata cho {len(midi_metadata)} file MIDI")
    
    # Step 2: Thu thập mô tả văn bản
    logger.info("\n📝 Step 2: Đang thu thập mô tả văn bản...")
    text_collector = OptimizedTextCollector(
        checkpoint_interval=checkpoint_interval,
        delay=delay
    )
    paired_data = text_collector.collect_text_for_all_midi(
        midi_metadata,
        checkpoint_dir=output_dir
    )
    
    paired_file = os.path.join(output_dir, "paired_data.json")
    with open(paired_file, "w", encoding="utf-8") as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Đã ghép cặp {len(paired_data)} file MIDI với mô tả văn bản")
    
    # Step 3: Tạo bộ dữ liệu hoàn chỉnh
    logger.info("\n🔗 Step 3: Đang tạo bộ dữ liệu hoàn chỉnh...")
    
    # Khởi tạo data_pairing
    if midi_metadata and "file_path" in midi_metadata[0]:
        midi_dir = os.path.dirname(midi_metadata[0]["file_path"])
    else:
        midi_dir = "data/midi"
    
    data_pairing = DataPairing(midi_dir)
    
    # Lọc dữ liệu nếu cần
    if filter_quality:
        filtered_dataset = data_pairing.filter_paired_data(
            paired_data, min_text_length=min_text_length, min_duration=min_duration
        )
        
        complete_file = os.path.join(output_dir, "complete_dataset_filtered.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)
        
        dataset = filtered_dataset
    else:
        complete_file = os.path.join(output_dir, "complete_dataset.json")
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(paired_data, f, indent=2, ensure_ascii=False)
        
        dataset = paired_data
    
    # Step 4: Validate dataset
    logger.info("\n✅ Step 4: Đang xác thực bộ dữ liệu...")
    stats = data_pairing.validate_paired_data(dataset)
    
    logger.info("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Tính thời gian xử lý
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    logger.info(f"\n🎉 Thu thập dữ liệu hoàn tất sau {time_str}!")
    logger.info("📁 Output files:")
    logger.info(f"  - MIDI metadata: {metadata_file}")
    logger.info(f"  - Paired data: {paired_file}")
    logger.info(f"  - Complete dataset: {complete_file}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Tiếp tục thu thập dữ liệu từ metadata với tối ưu")
    parser.add_argument("--metadata_file", type=str, default="data/output/midi_metadata.json", 
                        help="File metadata MIDI")
    parser.add_argument("--output_dir", type=str, default="data/output", 
                        help="Thư mục output")
    parser.add_argument("--checkpoint_interval", type=int, default=10, 
                        help="Số lượng file xử lý giữa mỗi lần lưu checkpoint")
    parser.add_argument("--delay", type=float, default=0.0, 
                        help="Thời gian chờ giữa các request (giây)")
    parser.add_argument("--filter_quality", action="store_true", 
                        help="Lọc dữ liệu theo chất lượng")
    parser.add_argument("--min_text_length", type=int, default=20, 
                        help="Độ dài tối thiểu của văn bản")
    parser.add_argument("--min_duration", type=float, default=10.0, 
                        help="Thời lượng tối thiểu của MIDI")
    parser.add_argument("--log_level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Mức độ log")
    
    args = parser.parse_args()
    
    continue_from_metadata(
        args.metadata_file, 
        args.output_dir,
        args.checkpoint_interval,
        args.delay,
        args.filter_quality,
        args.min_text_length,
        args.min_duration,
        args.log_level
    )


if __name__ == "__main__":
    main()