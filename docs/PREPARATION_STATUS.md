# Preparation status

Updated 26 November 2026. This revision extends the public code snapshot with
portable workflows. Archived research results keep their original provenance.

| Component | Status |
|---|---|
| Recorded implementations and original licenses | Preserved with source hashes |
| Re-DocRED reference results and intervals | Included |
| DWIE fixed split IDs and relation vocabulary | Included |
| Complete-download verification | `python -B verify_repository.py` |
| Re-DocRED C/B/D/G/U, C-off and averaging | Portable interfaces implemented; CPU contracts pass |
| DWIE preparation, C/B/D/G, evaluation and timing | Portable interfaces implemented; CPU data/CLI contracts pass |
| Symmetric Dev calibration and Re-DocRED timing | Portable interfaces implemented |
| Source pretraining from scratch | Historical runners preserved; no independent portable command provided |
| Checkpoint catalog | 18 identities in `metadata/checkpoints.json` |
| Checkpoint file re-verification | Six initial-release states rehashed on the research server and matched |
| Research-runtime CPU checks | 27 contracts and 9 critical module imports passed; model loading/CUDA initialization blocked |
| Checkpoint hosting | Hugging Face staging tools prepared; public URLs pending |
| Actual model execution through the new interfaces | Deferred until a GPU environment is available |

Command and data contracts exercise interfaces and conversion behavior without
loading pretrained weights. They do not establish agreement of model outputs.
The pending execution check is evaluation of existing checkpoints on Dev;
retraining and further Test evaluation are not needed for that check.

New timing commands record their own measurement order and boundaries. Their
measurements are distinct from the paper's archived timing sessions.
