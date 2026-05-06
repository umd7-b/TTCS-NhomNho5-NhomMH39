"""Script to strip all # comments from Python files."""
import os

def strip_comments(source):
    lines = source.splitlines(True)
    result = []
    in_triple_dq = False
    in_triple_sq = False

    for line in lines:
        stripped = line.lstrip()

        # Track triple-quoted strings across lines
        count_tdq = line.count('"""')
        count_tsq = line.count("'''")
        if count_tdq % 2 == 1:
            in_triple_dq = not in_triple_dq
        if count_tsq % 2 == 1:
            in_triple_sq = not in_triple_sq

        # If inside a triple-quoted string, keep line as-is
        if in_triple_dq or in_triple_sq:
            result.append(line)
            continue

        # Skip full-line comments
        if stripped.startswith('#'):
            continue

        # Remove inline comments
        in_sq = False
        in_dq = False
        i = 0
        while i < len(line):
            c = line[i]
            # Check for triple quotes first
            if line[i:i+3] == '"""':
                i += 3
                continue
            if line[i:i+3] == "'''":
                i += 3
                continue
            if c == '"' and not in_sq:
                in_dq = not in_dq
            elif c == "'" and not in_dq:
                in_sq = not in_sq
            elif c == '#' and not in_sq and not in_dq:
                code_part = line[:i].rstrip()
                if code_part:
                    result.append(code_part + '\n')
                break
            i += 1
        else:
            result.append(line)

    # Remove excessive blank lines
    text = ''.join(result)
    while '\n\n\n\n' in text:
        text = text.replace('\n\n\n\n', '\n\n')
    return text


files = [
    'plate_settings.py',
    'plate_models.py',
    'plate_features.py',
    'train.py',
    'train_ocr.py',
    'gui_plate_detector.py',
]

for f in files:
    if not os.path.exists(f):
        print(f'SKIP: {f}')
        continue
    with open(f, 'r', encoding='utf-8') as fh:
        original = fh.read()
    cleaned = strip_comments(original)
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(cleaned)
    orig_lines = len(original.splitlines())
    new_lines = len(cleaned.splitlines())
    print(f'{f}: {orig_lines} -> {new_lines} lines (removed {orig_lines - new_lines})')

print('Done!')
