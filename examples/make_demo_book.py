# /// script
# requires-python = ">=3.10"
# dependencies = ["pymupdf>=1.24"]
# ///
"""A small made-up demo book, to try the whole pipeline without a real book.

    uv run examples/make_demo_book.py demo

writes demo/Beekeeping Basics.pdf (digital, with a text layer) and
demo/Beekeeping Basics (scan).pdf (the same pages as images only, which goes through OCR).
"""
import sys
from pathlib import Path

import pymupdf

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
out_dir.mkdir(parents=True, exist_ok=True)
CSS = """
* {font-family: sans-serif;}
p {font-size: 10.5pt; line-height: 1.45; margin: 0 0 7pt 0; text-align: justify;}
h1 {font-size: 24pt; margin: 0 0 14pt 0;}
h2 {font-size: 14pt; margin: 12pt 0 6pt 0;}
li {font-size: 10.5pt; line-height: 1.45;}
.cap {font-size: 9.5pt; font-weight: bold;}
"""
P = lambda t: f"<p>{t}</p>"
pages = [
 ("", None, "<h1>Beekeeping Basics</h1><p>A short practical guide for new beekeepers.</p>"),
 ("", None, "<h1>Contents</h1><p>1 The Honey Bee Colony ........ 3</p><p>2 Hive Management ........ 6</p>"),
 ("1 The Honey Bee Colony", 3, "<h1>Chapter 1: The Honey Bee Colony</h1>"
  + P("A honey bee colony behaves like a single organism. Tens of thousands of workers, a few hundred drones and one queen share the work of building comb, raising brood and storing food for the winter. None of them can survive alone for long.")
  + "<h2>1.1 The queen</h2>"
  + P("The queen is the only fertile female in the colony. During the main season she can lay up to two thousand eggs per day, which is more than her own body weight. Her pheromones hold the colony together and signal to the workers that a laying queen is present.")
  + P("A queen usually lives for three to five years, but most beekeepers replace her after two seasons, because egg laying declines with age and an old queen makes swarming more likely.")
  + "<h2>1.2 Workers and drones</h2>"
  + P("Workers are sterile females. Their tasks change with age: young bees clean cells and feed larvae, middle-aged bees build comb and guard the entrance, and the oldest bees fly out as foragers to collect nectar, pollen, water and propolis.")),
 ("1 The Honey Bee Colony", 4, P("A summer worker lives only about six weeks, while winter bees can survive for several months. Drones are male bees whose only task is to mate with young queens from other colonies; they are expelled from the hive in late summer when food becomes scarce.")
  + "<div style='border:1px solid #333; height:150pt; margin:8pt 0 4pt 0; background:#eef3f8'></div>"
  + "<p class='cap'>Figure 1.1: Roles of queen, workers and drones in the colony</p>"
  + "<h2>1.3 Communication</h2>"
  + P("Foragers report rich food sources with the waggle dance. The angle of the dance on the vertical comb shows the direction of the source relative to the sun, and the duration of the waggle run shows the distance.")),
 ("1 The Honey Bee Colony", 5, P("Besides dancing, bees communicate through scent. The alarm pheromone, which smells of bananas, recruits guards to defend the entrance, and the Nasonov gland releases an orientation scent that guides lost bees home.")
  + P("Understanding these signals helps the beekeeper: a calm colony hums evenly, while a loud, high-pitched roar when the hive is opened can mean that the colony has lost its queen.")),
 ("2 Hive Management", 6, "<h1>Chapter 2: Hive Management</h1>"
  + P("Good hive management means looking into the colony regularly, but not too often. Every inspection disturbs the bees, cools the brood nest and costs the colony a day of work.")
  + "<h2>2.1 Hive inspections</h2>"
  + P("In spring and summer a colony should be inspected every seven to nine days. During each inspection the beekeeper checks the following points:")
  + "<ul><li>Is there fresh brood, and are eggs standing upright in the cells?</li><li>Is the brood pattern compact or patchy?</li><li>Are there queen cells along the bottom edge of the frames?</li><li>Do the bees have enough stores of honey and pollen?</li><li>Are there signs of disease or varroa mites?</li></ul>"
  + P("Work slowly, use a little smoke at the entrance, and keep each frame above the open hive so that the queen cannot fall into the grass.")),
 ("2 Hive Management", 7, "<h2>2.2 Preventing swarms</h2>"
  + P("Swarming is the natural way a colony reproduces: the old queen leaves with about half of the workers to found a new home. For the beekeeper a swarm means losing bees and most of the honey harvest.")
  + P("The most reliable ways to prevent swarming are to give the colony enough space in time by adding a super, to remove queen cells during inspections, and to make an artificial swarm or a nucleus colony when the bees are clearly preparing to swarm.")
  + "<p class='cap'>Table 2.1: Inspection checklist for swarm prevention</p>"
  + P("Young queens swarm less often than old ones, which is another reason to requeen regularly.")),
 ("2 Hive Management", 8, "<h2>2.3 Harvesting honey</h2>"
  + P("Honey is ready to harvest when the bees have capped at least two thirds of the cells with wax. Its water content should be below eighteen percent; wetter honey ferments in the jar.")
  + P("Remove the supers with a bee escape board, uncap the combs with a fork or a heated knife, and spin the honey out in an extractor. Always leave the colony enough food for the winter, or feed sugar syrup in late summer.")),
]
doc = pymupdf.open()
for header, number, html in pages:
    page = doc.new_page(width=595, height=842)
    if header:
        page.insert_text((70, 32), header, fontsize=8.5, fontname="helv")
    page.insert_htmlbox(pymupdf.Rect(70, 70, 525, 780), html, css=CSS)
    if number:
        page.insert_text((290, 815), str(number), fontsize=9, fontname="helv")
doc.save(f"{out_dir}/Beekeeping Basics.pdf")
scan = pymupdf.open()
for page in doc:
    pix = page.get_pixmap(dpi=150)
    p = scan.new_page(width=page.rect.width, height=page.rect.height)
    p.insert_image(p.rect, pixmap=pix)
scan.save(f"{out_dir}/Beekeeping Basics (scan).pdf")
print(f"{doc.page_count} pages → {out_dir / 'Beekeeping Basics.pdf'} and {out_dir / 'Beekeeping Basics (scan).pdf'}")
