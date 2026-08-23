# Reference floors

How much a model's output moves when nothing is compressed. Compression numbers
are only interpretable against this.

| File | Floor |
|---|---|
| `score_sampling_floor.py` | second L0 draw at T=0.7 vs the greedy L0 |
| `score_register_floor.py` | L0 under a register change, no compression |
| `score_length_controlled.py` | rescoring with length held fixed |

Generate the paired draw first (`--temperature 0.7 --suffix _paired` on the
relevant runner), then:

```bash
python score_sampling_floor.py          # --force to rescan finished cells
```

Reads `results/<model>_<condition>/<dataset>/`, writes one summary per cell to
`results/revision/paired_l0/`, skips cells already done.

Floors are task-specific: long-form numbers need the long-form floor. There is
no sampling floor for API models, since it requires control over decoding.
