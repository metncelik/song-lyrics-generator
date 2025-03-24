#!/usr/bin/env python
import argparse
from inference import generate_lyrics

def main():
    parser = argparse.ArgumentParser(description='Generate song lyrics from a prompt')
    
    parser.add_argument('prompt', type=str, help='Initial prompt for lyrics generation')
    parser.add_argument('--max-length', type=int, default=150, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top-k', type=int, default=75, help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling parameter')
    
    args = parser.parse_args()
    
    print(f"\nGenerating lyrics from prompt: '{args.prompt}'")
    print("-" * 50)
    
    lyrics = generate_lyrics(
        args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print(lyrics)
    print("\n" + "-" * 50)

if __name__ == "__main__":
    main() 