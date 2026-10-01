# Attribution and external artifacts

`ChessFlyStatePings` is an experimental derivative/research harness around **ChessFly** by Maxime Labonne (`mlabonne/chessfly` on Hugging Face). The original ChessFly project, checkpoint, Space assets, model card, and their cited connectome sources remain upstream work and are not authored by this repository.

The large external files `flynet.safetensors`, `connectome.bin.gz`, and `neurons.bin.gz` are not redistributed here. The runtime obtains them directly from the original Hugging Face locations and records their hashes. Upstream terms and the FlyWire-derived data terms cited by ChessFly remain authoritative for those artifacts.

The StatePing experiment is motivated by `anttiluode/BrainAsInverseModelerV3`, especially the distinction between retained internal state, emitted state, and receiver-dependent readout.

Repository source code is GPL-3.0-only unless a file says otherwise.
