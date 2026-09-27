"""Detect frame content that runs under the green footer rule.

The template's footline is a *fixed* picture: the same name/chair text, logo and
rule on every page.  So take the pixels that are dark on EVERY page as the
reference footer, and report any page whose dark pixels below the rule are not
explained by that reference (the page number, which changes, is excluded by
looking only at the text-block columns).

Usage: python footer_check.py <render-dir>
"""
import sys, glob, os
import numpy as np
from PIL import Image

render_dir = sys.argv[1] if len(sys.argv) > 1 else "report_slides/render"
pages = sorted(glob.glob(os.path.join(render_dir, "pg-*.png")))
if not pages:
    sys.exit("no renders in " + render_dir)

imgs = [np.asarray(Image.open(p).convert("RGB")).astype(int) for p in pages]
h, w, _ = imgs[0].shape
if any(a.shape != imgs[0].shape for a in imgs):
    sys.exit("renders differ in size")

r, g, b = imgs[-1][:, :, 0], imgs[-1][:, :, 1], imgs[-1][:, :, 2]
greenish = (g > 100) & (g > r + 30) & (g > b + 30)
rows = [y for y in range(h) if greenish[y].sum() > 0.8 * w]
rule_top, rule_bot = min(rows), max(rows) + 2
# Start the scan a few rows ABOVE the rule: a frame whose last line merely clips
# the rule has most of its ink in the band, not below it, and starting at
# rule_bot misses exactly that case.
scan_from = rule_top - 6
print(f"image {w}x{h}   footer rule rows {rule_top}..{rule_bot}, scanning from y={scan_from}")

def dark(a):
    return (a[:, :, 0] < 150) & (a[:, :, 1] < 150) & (a[:, :, 2] < 150)

# fixed footer = dark on every page that HAS a footer.  Page 1 is the title
# slide, which suppresses the footline entirely, so it must not vote.
ref = np.ones((h, w), bool)
for a in imgs[1:]:
    ref &= dark(a)

# text-block columns only: the footer name/chair sits far left, logo far right
x0, x1 = int(0.055 * w), int(0.88 * w)
bad = []
for p, a in zip(pages, imgs):
    extra = dark(a) & ~ref
    extra[:scan_from, :] = False         # the rule band and everything below it
    extra[:, :x0] = False
    extra[:, x1:] = False
    n = int(extra.sum())
    if n > 40:
        bad.append((os.path.basename(p), n))
        print(f"  {os.path.basename(p):10s} ink below rule not in footer: {n} px   <-- OVERFLOW")

if not bad:
    print("clean: no body text below the footer rule on any page")
