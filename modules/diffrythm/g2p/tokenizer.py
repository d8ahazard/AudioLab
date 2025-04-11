"""
Tokenizer for DiffRhythm

Simplified version of the original DiffRhythm tokenizer
"""

import json
import os


class CNENTokenizer:
    """
    Chinese and English Tokenizer for DiffRhythm

    Simple implementation that handles text without the full g2p dependency
    """
    def __init__(self):
        # Define a default vocab for basic usage
        self.phone2id = {str(i): i for i in range(363)}  # Default vocab size in DiffRhythm
        self.id2phone = {v: k for k, v in self.phone2id.items()}
        
        # Try to load the real vocab file if it exists
        vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    self.phone2id = data["vocab"]
                    self.id2phone = {v: k for k, v in self.phone2id.items()}
            except Exception as e:
                print(f"Warning: Could not load vocab file: {e}")
    
    def encode(self, text):
        """
        Encode text to token IDs

        In the absence of the full g2p system, this provides a simplified encoding
        that maps characters to a range of token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Simple character-based encoding for basic functionality
        # In a real implementation, this would use the g2p system
        result = []
        for char in text:
            # Map each character to a token ID based on its unicode value
            # This is just a fallback when the real g2p system isn't available
            char_val = ord(char) % 360  # Modulo to keep within vocab range
            if char_val == 0:  # Avoid 0 which is usually reserved for padding
                char_val = 1
            result.append(char_val)
        return result
    
    def decode(self, tokens):
        """
        Decode token IDs to text
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        return "|".join([self.id2phone.get(x - 1, "#") for x in tokens]) 