#!/usr/bin/env python3
"""
Tag Validator for Daily LLM Papers
Validates paper tags against approved list and ensures consistency.
"""

import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from difflib import get_close_matches
from collections import Counter
import yaml

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Approved tags by category
APPROVED_TAGS = {
    'technical': {
        # Architecture
        'transformer', 'state-space', 'mixture-of-experts', 'moe', 'diffusion', 
        'gan', 'hybrid', 'attention', 'mamba', 'rwkv', 'retnet',
        
        # Training Methods
        'rlhf', 'dpo', 'supervised', 'self-supervised', 'constitutional',
        'instruction-tuning', 'few-shot', 'zero-shot', 'in-context-learning',
        'reinforcement-learning', 'preference-optimization',
        
        # Optimization
        'quantization', '4-bit', '8-bit', 'pruning', 'distillation', 'lora',
        'qlora', 'peft', 'adapter', 'flash-attention', 'sparse', 'efficient',
        
        # Capabilities
        'reasoning', 'chain-of-thought', 'cot', 'tool-use', 'code-generation',
        'math', 'multimodal', 'vision', 'audio', 'video', 'long-context',
        '32k-context', '100k-context', '1m-context', 'planning', 'agent',
        'retrieval', 'rag', 'memory', 'knowledge',
    },
    
    'impact': {
        'sota', 'breakthrough', 'incremental', 'competitive', 'novel-architecture',
        'novel-method', 'novel-application', 'benchmark', 'analysis',
        'production-ready', 'experimental', 'theoretical', 'reproducible',
        'influential', 'foundational',
    },
    
    'meta': {
        # Organizations
        'openai', 'anthropic', 'google', 'meta', 'microsoft', 'deepmind',
        'mistral', 'stability-ai', 'cohere', 'academic', 'independent',
        'stanford', 'mit', 'berkeley', 'oxford', 'cambridge',
        
        # Model Size
        '<1b', '1b-7b', '7b-30b', '30b-100b', '100b+', 'size-unknown',
        
        # License/Availability
        'open-source', 'open-weights', 'api-only', 'proprietary', 'closed-source',
        'mit', 'apache-2.0', 'cc-by', 'commercial',
        
        # Resource Requirements
        'consumer-gpu', 'single-gpu', 'multi-gpu', 'datacenter', 'edge-device',
        'cpu-inference', 'mobile', 'cloud-only',
        
        # Time
        '2023', '2024', '2025', 'recent', 'classic',
    }
}

# Flatten all approved tags for quick lookup
ALL_APPROVED_TAGS = set()
for category_tags in APPROVED_TAGS.values():
    ALL_APPROVED_TAGS.update(category_tags)

# Common tag mistakes and their corrections
TAG_CORRECTIONS = {
    'llm': None,  # Too generic, remove
    'ai': None,   # Too generic, remove
    'gpt': 'transformer',
    'bert': 'transformer',
    'large-model': '100b+',
    'small-model': '<1b',
    'efficient-training': 'efficient',
    'vision-language': 'multimodal',
    'prompt-engineering': 'prompting',
    'prompt': 'prompting',
    'fine-tune': 'fine-tuning',
    'finetune': 'fine-tuning',
}

class TagValidator:
    def __init__(self, repo_path: Path = Path('.')):
        self.repo_path = repo_path
        self.errors = []
        self.warnings = []
        self.stats = Counter()
        
    def extract_tags_from_file(self, filepath: Path) -> Dict[str, List[str]]:
        """Extract all tags from a markdown file."""
        papers_with_tags = {}
        
        try:
            content = filepath.read_text(encoding='utf-8')
            
            # Find all paper entries with tags
            # Pattern: **Tags:** `tag1` `tag2` `tag3`
            pattern = r'\*\*Tags:\*\*\s*((?:`[^`]+`\s*)+)'
            matches = re.finditer(pattern, content, re.DOTALL)
            
            for match in matches:
                tags_str = match.group(1)
                tags = re.findall(r'`([^`]+)`', tags_str)
                # For now, use a generic paper title since we're just extracting tags
                paper_title = f"Paper_{len(papers_with_tags) + 1}"
                papers_with_tags[paper_title] = tags
                
        except Exception as e:
            self.errors.append(f"Error reading {filepath}: {e}")
            
        return papers_with_tags
    
    def validate_tags(self, tags: List[str], paper_title: str) -> Tuple[List[str], List[str]]:
        """Validate a list of tags for a paper."""
        invalid_tags = []
        suggestions = {}
        
        for tag in tags:
            tag_lower = tag.lower().strip()
            
            # Check if tag needs correction
            if tag_lower in TAG_CORRECTIONS:
                correction = TAG_CORRECTIONS[tag_lower]
                if correction:
                    suggestions[tag] = correction
                else:
                    invalid_tags.append(tag)
                continue
            
            # Check if tag is approved
            if tag_lower not in ALL_APPROVED_TAGS:
                # Find close matches
                close_matches = get_close_matches(tag_lower, ALL_APPROVED_TAGS, n=3, cutoff=0.6)
                if close_matches:
                    suggestions[tag] = close_matches[0]
                invalid_tags.append(tag)
        
        return invalid_tags, suggestions
    
    def check_tag_rules(self, tags: List[str], paper_title: str):
        """Check if tags follow best practices."""
        # Rule 1: Should have 2-5 tags
        if len(tags) < 2:
            self.warnings.append(f"⚠️  '{paper_title}': Too few tags ({len(tags)}), recommend 2-5")
        elif len(tags) > 5:
            self.warnings.append(f"⚠️  '{paper_title}': Too many tags ({len(tags)}), recommend 2-5")
        
        # Rule 2: Should have mix of tag types
        tag_categories = {'technical': 0, 'impact': 0, 'meta': 0}
        for tag in tags:
            tag_lower = tag.lower()
            for category, category_tags in APPROVED_TAGS.items():
                if tag_lower in category_tags:
                    tag_categories[category] += 1
                    break
        
        if sum(1 for count in tag_categories.values() if count > 0) < 2:
            self.warnings.append(f"⚠️  '{paper_title}': Tags lack diversity (use technical + impact/meta)")
        
        # Rule 3: Check for redundant tags
        if 'efficient' in tags and 'optimization' in tags:
            self.warnings.append(f"⚠️  '{paper_title}': Redundant tags (efficient + optimization)")
    
    def analyze_tag_usage(self, all_tags: List[str]):
        """Generate statistics about tag usage."""
        tag_counter = Counter(tag.lower() for tag in all_tags)
        
        print(f"\n{Colors.BLUE}📊 Tag Usage Statistics{Colors.END}")
        print("=" * 50)
        
        # Most used tags
        print(f"\n{Colors.BOLD}Top 10 Most Used Tags:{Colors.END}")
        for tag, count in tag_counter.most_common(10):
            bar = '█' * min(count, 20)
            print(f"  {tag:20} {bar} ({count})")
        
        # Tag distribution by category
        print(f"\n{Colors.BOLD}Tag Distribution by Category:{Colors.END}")
        category_counts = {'technical': 0, 'impact': 0, 'meta': 0}
        for tag, count in tag_counter.items():
            for category, tags in APPROVED_TAGS.items():
                if tag in tags:
                    category_counts[category] += count
                    break
        
        total = sum(category_counts.values())
        for category, count in category_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {category:12} {count:4} tags ({percentage:.1f}%)")
    
    def validate_repository(self):
        """Validate all markdown files in the repository."""
        print(f"{Colors.BOLD}🏷️  Tag Validator for Daily LLM Papers{Colors.END}")
        print("=" * 50)
        
        # Files to check
        files_to_check = [self.repo_path / 'README.md']
        
        # Add category files if they exist
        categories_dir = self.repo_path / 'categories'
        if categories_dir.exists():
            files_to_check.extend(list(categories_dir.glob('*.md')))
        
        all_tags = []
        total_papers = 0
        
        for filepath in files_to_check:
            if not filepath.exists():
                continue
                
            print(f"\n{Colors.BLUE}Checking {filepath.name}...{Colors.END}")
            papers_with_tags = self.extract_tags_from_file(filepath)
            
            for paper_title, tags in papers_with_tags.items():
                total_papers += 1
                all_tags.extend(tags)
                
                # Validate tags
                invalid_tags, suggestions = self.validate_tags(tags, paper_title)
                
                if invalid_tags:
                    print(f"\n{Colors.RED}❌ Invalid tags in '{paper_title}':{Colors.END}")
                    for tag in invalid_tags:
                        if tag in suggestions:
                            print(f"   '{tag}' → suggest: '{Colors.GREEN}{suggestions[tag]}{Colors.END}'")
                        else:
                            print(f"   '{tag}' → {Colors.YELLOW}no suggestion found{Colors.END}")
                
                # Check tag rules
                self.check_tag_rules(tags, paper_title)
        
        # Print warnings
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠️  Warnings:{Colors.END}")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Print statistics
        self.analyze_tag_usage(all_tags)
        
        # Summary
        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total papers checked: {total_papers}")
        print(f"  Total tags found: {len(all_tags)}")
        print(f"  Unique tags: {len(set(tag.lower() for tag in all_tags))}")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n{Colors.RED}❌ Validation failed with {len(self.errors)} errors{Colors.END}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ All tags validated successfully!{Colors.END}")
            return True
    
    def generate_tag_report(self, output_file: Path = Path('tag-report.json')):
        """Generate a detailed JSON report of all tags."""
        report = {
            'approved_tags': APPROVED_TAGS,
            'total_approved': len(ALL_APPROVED_TAGS),
            'validation_errors': self.errors,
            'validation_warnings': self.warnings,
            'statistics': dict(self.stats)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to {output_file}")

def main():
    """Main entry point for the script."""
    validator = TagValidator()
    
    # Run validation
    success = validator.validate_repository()
    
    # Generate report if requested
    if '--report' in sys.argv:
        validator.generate_tag_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()