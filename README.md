# The Human Spark — a Remotion film

A 45-second, editorial-style animated timeline of human innovation, built entirely with React and Remotion. The composition moves from early stone tools through agriculture, print, industry, electricity, and networked computing, ending with a question about how we choose to use emerging technologies.

## Run it

```bash
npm install
npm start
```

Render the film or a representative still:

```bash
npm run render
npm run still
```

The `HumanInnovation` composition is 1920×1080, 30 fps, and 45 seconds long. All visuals are procedural HTML/SVG, so the project has no external asset or licensing dependencies.

The render command uses a packaged Chromium executable, avoiding Remotion's usual first-run browser download and making rendering reliable in restricted or offline build environments.

## Editorial approach

This is a thematic history, not a claim that invention followed one straight, inevitable line. It emphasizes three recurring ideas:

1. **Innovation is cumulative.** Tools become platforms for later tools.
2. **Innovation is collective.** The story resists “lone genius” mythology and foregrounds transmission and networks.
3. **Innovation has trade-offs.** Agriculture and industry created abundance while also amplifying hierarchy, exploitation, and environmental damage.

Dates and framing were checked against the following institutional sources:

- [Smithsonian Human Origins — Early Stone Age Tools](https://humanorigins.si.edu/evidence/behavior/stone-tools/early-stone-age-tools) (earliest stone-tool evidence at least 2.6 million years ago; newer Lomekwi finds are dated to 3.3 million years).
- [British Museum — A History of the World: Farming](https://www.britishmuseum.org/collection/galleries/early-farming) (the shift toward settled farming from roughly 10,000 BCE).
- [Encyclopaedia Britannica — History of Printing](https://www.britannica.com/topic/printing-publishing/History-of-printing) (European movable-type printing and Gutenberg in the mid-15th century).
- [Science Museum — The Industrial Revolution](https://www.sciencemuseum.org.uk/objects-and-stories/industrial-revolution) (steam, factories, transport, and the social effects of industrialization).
- [CERN — The Birth of the Web](https://home.cern/science/computing/birth-web) (Tim Berners-Lee's 1989 proposal and the open web).

## Structure

- `src/Root.tsx` defines the composition.
- `src/HumanInnovation.tsx` contains the nine timed scenes.
- `src/components.tsx` provides reusable motion and visual primitives.
- `src/theme.ts` owns the deliberately limited editorial palette.
