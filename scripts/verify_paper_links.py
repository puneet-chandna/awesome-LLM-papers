#!/usr/bin/env python3
"""
Paper Title Verification Tool v2
Verifies that paper links point to the correct papers based on title matching.
Now catches papers in tables, inline links, and various markdown formats.

Usage:
    python verify_paper_links.py .                    # Check all markdown files
    python verify_paper_links.py categories/          # Check specific directory
    python verify_paper_links.py README.md -t 0.9     # Stricter threshold
"""

import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urlparse

try:
    import arxiv
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install --break-system-packages arxiv beautifulsoup4 requests")
    sys.exit(1)


@dataclass
class PaperLink:
    """Represents a paper link from markdown."""
    expected_title: str
    url: str
    file: str
    line: int


@dataclass
class VerificationResult:
    """Result of verifying a paper link."""
    paper: PaperLink
    actual_title: Optional[str]
    similarity: float
    status: str  # "verified", "mismatch", "error", "skipped"
    error: Optional[str] = None


# Multiple patterns to catch different markdown link formats
PATTERNS = [
    # ### [Title](URL) or ## [Title](URL) - heading style
    re.compile(r'^#{2,4}\s*(?:📄\s*)?(?:🆕\s*)?\[([^\]]+)\]\(([^)]+)\)', re.MULTILINE),
    # **[Title](URL)** - bold style
    re.compile(r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*'),
    # | [Title](URL) | - table cell (but NOT icons like [📄])
    re.compile(r'\|\s*\[([A-Za-z][^\]]{5,})\]\(([^)]+)\)'),
]

# URLs to skip (not papers - badges, icons, internal links, etc.)
SKIP_PATTERNS = [
    r'img\.shields\.io',
    r'github\.com/[^/]+/[^/]+/issues',
    r'github\.com/[^/]+/[^/]+/graphs',
    r'github\.com/[^/]+/[^/]+/stargazers',
    r'github\.com/[^/]+$',  # Just org/user links
    r'awesome\.re',
    r'^#',  # Internal anchors
    r'\.png$',
    r'\.jpg$',
    r'\.svg$',
    r'twitter\.com',
    r'x\.com/\w+/status',
    r'discord\.gg',
    r'promptingguide\.ai',
    r'learnprompting\.org',
    r'gpt3demo\.com',
    r'jalammar\.github\.io/illustrated',
    r'huggingface\.co/blog',
    r'huggingface\.co/docs',
    r'opensource\.org',
    r'research\.google/?$',
]

# Valid paper URL patterns
PAPER_URL_PATTERNS = [
    r'arxiv\.org',
    r'openreview\.net',
    r'aclanthology\.org',
    r'proceedings\.neurips',
    r'proceedings\.mlr\.press',
    r'nature\.com',
    r'science\.org',
    r'ieee\.org',
    r'acm\.org',
    r'anthropic\.com.*\.pdf',
    r'openai\.com',
    r'cdn\.openai\.com',
    r'deepmind\.com',
    r'ai\.google',
    r'research\.google',
    r'transformer-circuits\.pub',
    r'ml-site\.cdn-apple\.com',
    r'llava-vl\.github\.io',
]


def is_paper_url(url: str) -> bool:
    """Check if URL is likely a paper link."""
    for pattern in PAPER_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def extract_paper_links(markdown_path: Path) -> list[PaperLink]:
    """Extract paper title and URL pairs from markdown file."""
    links = []
    content = markdown_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    seen_urls = set()  # Deduplicate
    
    for line_num, line in enumerate(lines, 1):
        for pattern in PATTERNS:
            for match in pattern.finditer(line):
                title, url = match.groups()
                
                # Clean up title
                title = title.strip()
                title = re.sub(r'^[🧮🧠👁️🌍⚡🏗️🤖📄🆕\s]+', '', title)  # Remove leading emojis
                title = title.strip('*').strip()  # Remove bold markers
                
                # Skip if not a paper URL or should be skipped
                if should_skip_url(url):
                    continue
                if not is_paper_url(url):
                    continue
                    
                # Skip very short titles (likely icons)
                if len(title) < 10:
                    continue
                
                # Deduplicate
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                links.append(PaperLink(
                    expected_title=title,
                    url=url.strip(),
                    file=str(markdown_path),
                    line=line_num
                ))
    
    return links


def get_arxiv_title(arxiv_url: str) -> Optional[str]:
    """Get paper title from arXiv using official API."""
    # Extract arXiv ID from URL (handle both abs and pdf formats)
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', arxiv_url)
    if not match:
        # Try older format: arxiv.org/abs/hep-th/9901001
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([a-z-]+/\d+)', arxiv_url)

    if not match:
        return None

    arxiv_id = match.group(1).replace('.pdf', '')

    try:
        # Rate limit: arXiv requires 3 second delay between requests
        time.sleep(3)
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        return paper.title
    except StopIteration:
        print(f"not found", end=" ")
        return None
    except Exception as e:
        print(f"API error", end=" ")
        return None


def get_web_page_title(url: str) -> Optional[str]:
    """Get paper title by scraping web page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        time.sleep(1)  # Be polite
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try og:title meta tag first (most reliable for paper pages)
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        # Try twitter:title
        tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw_title and tw_title.get('content'):
            return tw_title['content'].strip()

        # Fall back to <title> tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        # Try h1 tag
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()

        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"403", end=" ")
        else:
            print(f"HTTP {e.response.status_code}", end=" ")
        return None
    except Exception as e:
        print(f"error", end=" ")
        return None


def get_actual_title(url: str) -> Optional[str]:
    """Get the actual paper title from URL."""
    if 'arxiv.org' in url:
        return get_arxiv_title(url)
    else:
        return get_web_page_title(url)


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Remove common suffixes/prefixes
    title = re.sub(r'\s*\|\s*.*$', '', title)  # Remove " | Site Name"
    title = re.sub(r'\s*-\s*arXiv.*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^arXiv:\d+\.\d+\s*', '', title)
    title = re.sub(r'\s*\[.*?\]\s*', ' ', title)  # Remove [v1], [PDF], etc.
    title = re.sub(r'\s*\(.*?\)\s*$', '', title)  # Remove trailing (Author, Year)
    title = re.sub(r'\s+', ' ', title).strip()
    return title.lower()


def calculate_similarity(expected: str, actual: str) -> float:
    """Calculate similarity between expected and actual titles."""
    expected_norm = normalize_title(expected)
    actual_norm = normalize_title(actual)
    
    # Direct ratio
    ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    
    # Also check if one is a substring of the other (common for abbreviated titles)
    if expected_norm in actual_norm or actual_norm in expected_norm:
        ratio = max(ratio, 0.85)
    
    # Check if key words match
    expected_words = set(expected_norm.split())
    actual_words = set(actual_norm.split())
    common_words = expected_words & actual_words
    
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'of', 'for', 'and', 'in', 'on', 'with', 'to', 'from', 'via', 'are', 'is'}
    key_expected = expected_words - stop_words
    key_actual = actual_words - stop_words
    key_common = key_expected & key_actual
    
    if key_expected and len(key_common) / len(key_expected) > 0.7:
        ratio = max(ratio, 0.80)
    
    return ratio


def verify_paper(paper: PaperLink, threshold: float = 0.8) -> VerificationResult:
    """Verify a single paper link."""
    truncated_title = paper.expected_title[:45] + "..." if len(paper.expected_title) > 45 else paper.expected_title
    print(f"  📄 {truncated_title}", end=" → ", flush=True)

    actual_title = get_actual_title(paper.url)

    if actual_title is None:
        print("❌ (fetch failed)")
        return VerificationResult(
            paper=paper,
            actual_title=None,
            similarity=0.0,
            status="error",
            error="Could not fetch title from URL"
        )

    similarity = calculate_similarity(paper.expected_title, actual_title)

    if similarity >= threshold:
        print(f"✅ {similarity:.0%}")
        status = "verified"
    else:
        print(f"⚠️  MISMATCH {similarity:.0%}")
        status = "mismatch"

    return VerificationResult(
        paper=paper,
        actual_title=actual_title,
        similarity=similarity,
        status=status
    )


def generate_report(results: list[VerificationResult], output_path: Path):
    """Generate markdown report of verification results."""
    verified = [r for r in results if r.status == "verified"]
    mismatches = [r for r in results if r.status == "mismatch"]
    errors = [r for r in results if r.status == "error"]

    report = f"""# Paper Link Verification Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Status | Count |
|--------|-------|
| ✅ Verified | {len(verified)} |
| ⚠️ Mismatch | {len(mismatches)} |
| ❌ Error | {len(errors)} |
| **Total** | **{len(results)}** |

---

"""

    if mismatches:
        report += "## ⚠️ Mismatches (Requires Manual Review)\n\n"
        report += "> These links may point to the wrong papers. Review each one carefully.\n\n"
        
        # Sort by similarity - lowest first (most likely wrong)
        mismatches_sorted = sorted(mismatches, key=lambda r: r.similarity)
        
        for r in mismatches_sorted:
            report += f"""### {r.paper.expected_title}

| Field | Value |
|-------|-------|
| **File** | `{Path(r.paper.file).name}` (line {r.paper.line}) |
| **URL** | [{r.paper.url}]({r.paper.url}) |
| **Expected Title** | {r.paper.expected_title} |
| **Actual Title** | {r.actual_title} |
| **Similarity** | {r.similarity:.1%} |

---

"""

    if errors:
        report += "## ❌ Errors (Could Not Verify)\n\n"
        report += "These URLs could not be fetched (may be blocked, rate-limited, or offline):\n\n"
        report += "| File | Line | Expected Title | Error |\n"
        report += "|------|------|----------------|-------|\n"
        for r in errors:
            short_title = r.paper.expected_title[:40] + "..." if len(r.paper.expected_title) > 40 else r.paper.expected_title
            report += f"| `{Path(r.paper.file).name}` | {r.paper.line} | {short_title} | {r.error} |\n"
        report += "\n"

    if verified and not mismatches:
        report += "## ✅ All Papers Verified!\n\n"
        report += "All paper links point to the correct papers.\n"

    output_path.write_text(report, encoding='utf-8')
    print(f"\n📋 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify paper links point to correct papers based on title matching"
    )
    parser.add_argument(
        "paths", 
        nargs="+", 
        help="Markdown files or directories to check"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float, 
        default=0.8,
        help="Similarity threshold (0-1). Default: 0.8 (80%% match required)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="verification_report.md",
        help="Output report path. Default: verification_report.md"
    )
    parser.add_argument(
        "--arxiv-only",
        action="store_true",
        help="Only verify arXiv links (faster, more reliable)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("📚 Paper Title Verification Tool v2")
    print("=" * 60)
    print()

    # Collect all markdown files
    md_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.md':
            md_files.append(path)
        elif path.is_dir():
            md_files.extend(path.rglob('*.md'))
    
    # Exclude verification report itself
    md_files = [f for f in md_files if 'verification_report' not in f.name]

    print(f"🔍 Found {len(md_files)} markdown files\n")

    # Extract all paper links
    all_links = []
    for md_file in md_files:
        links = extract_paper_links(md_file)
        if args.arxiv_only:
            links = [l for l in links if 'arxiv.org' in l.url]
        all_links.extend(links)

    # Deduplicate by URL across all files
    seen = {}
    unique_links = []
    for link in all_links:
        if link.url not in seen:
            seen[link.url] = link
            unique_links.append(link)
    
    print(f"📚 Found {len(unique_links)} unique paper links to verify")
    if len(all_links) > len(unique_links):
        print(f"   (deduplicated from {len(all_links)} total links)")
    if args.arxiv_only:
        print("   (--arxiv-only mode: checking only arXiv links)")
    print()

    if not unique_links:
        print("No paper links found!")
        sys.exit(0)

    # Verify each link
    results = []
    for i, paper in enumerate(unique_links, 1):
        print(f"[{i}/{len(unique_links)}]", end="")
        result = verify_paper(paper, threshold=args.threshold)
        results.append(result)

    # Generate report
    generate_report(results, Path(args.output))

    # Summary
    verified = len([r for r in results if r.status == "verified"])
    mismatches = len([r for r in results if r.status == "mismatch"])
    errors = len([r for r in results if r.status == "error"])

    print()
    print("=" * 60)
    print(f"✅ Verified: {verified}  |  ⚠️ Mismatch: {mismatches}  |  ❌ Error: {errors}")
    print("=" * 60)

    # Exit with error if mismatches found
    if mismatches > 0:
        print(f"\n⚠️  Found {mismatches} potential mismatches! Review the report.")
        sys.exit(1)
    else:
        print("\n✅ All verified papers match their titles!")


if __name__ == "__main__":
    main()
