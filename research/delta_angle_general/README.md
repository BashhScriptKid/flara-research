# General Delta Angle Research

This folder contains the original research on delta angle as a general-purpose security measure against prompt injection.

## Key Findings

- Delta angle alone cannot reliably distinguish between normal and injection prompts
- Distributions overlap by >50%
- More valuable as auxiliary scaler than standalone detector
- Led to refinement: delta angle is effective for obfuscation detection (see `../obfuscation_detection/`)

## Files

- `Delta_Angle_Security_Paper.md` — Original research paper

## Relationship to Obfuscation Detection

The limitations discovered in this research led to the focused study on obfuscation detection, where delta angle performs significantly better.
