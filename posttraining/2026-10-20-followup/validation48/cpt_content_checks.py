"""Conservative, source-review-driven exclusions, never a quality approval."""
import re


def content_issue(text, domain):
    if domain == 'math':
        if re.match(r'^#\s+Questions tagged\s+\[', text) and text.count('\n### ') >= 3:
            return 'tag_index_with_incomplete_question_excerpts'
        if len(re.findall(r'\\(?:re)?newcommand\b', text)) >= 8:
            return 'excessive_latex_macro_boilerplate'
        if re.search(r'\b(?:graph|diagram|figure|table)\s+(?:below|above|shown)\b', text, re.I):
            return 'unavailable_visual_context'
    return None
