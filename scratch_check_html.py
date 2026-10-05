with open("experiments/results/ieee_report.html", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'class="page"', text)]
print(f"Total 'class=\"page\"' found: {len(matches)}")
for i, m in enumerate(matches):
    snippet = text[m:m+50]
    print(f"Page {i+1}: {snippet}")

print("\nLet's check where the text ends:")
print(text[-200:])
