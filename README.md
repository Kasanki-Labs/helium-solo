# Helium Solo: the open parts

Helium Solo records, replays, tests and fixes safe AI agents on your own machine. It is free, by Kasanki Labs: https://helium-solo.pages.dev

This repository holds the parts of Helium Solo that are open source:

- `sdk/`: the Kasanki Helium SDK (`pip install kasanki-helium-sdk`), which records AI agent runs. It is the code that sees your prompts, answers and keys, so you can check what it does with them.
- `spec/trace-format.md`: the format of Helium's records, and how to check that none of them was changed, removed or added.
- A verifier for those records, in plain Python with no dependencies: `python -m kasanki_helium.verify --data-dir <folder>`, in `sdk/src/kasanki_helium/verify.py`.

The rest of Helium Solo (the `helium` program, the desktop app and the `helium-solo` package) is free to use under its own terms and is not in this repository. To use Helium: `pip install helium-solo`.

## Licence

GNU Affero General Public License version 3 (LICENSE), with one additional term under its section 7(b): keep the attribution described in CREDIT.md.

## Contributing

See CONTRIBUTING.md. For security problems, see SECURITY.md.
