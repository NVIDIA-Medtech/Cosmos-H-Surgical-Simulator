# Security migration guide

## Released weights are Safetensors

Released model weights are distributed as `.safetensors`. Safetensors contains
no executable code, so loading it cannot run arbitrary code, and it is the
mandated format for distributed NVIDIA model weights. `--ckpt_path` accepts a
`.safetensors` file anywhere a single-file checkpoint is expected; `easy_io`
reads and writes the format by extension.

Legacy `.pt` checkpoints still load, under restricted loading (see below), so
existing local training outputs keep working. Prefer `.safetensors` for anything
you distribute. Convert a trusted `.pt` you produced yourself with
`safetensors.torch.save_file`, after confirming it holds only tensors.

## Checkpoints

Use a supported CUDA extra (PyTorch 2.7 or 2.9). Runtime checkpoint readers use
restricted loading and reject custom Python objects and unsafe loader overrides.
Model state dictionaries and ordinary optimizer state dictionaries are supported.
Generic distributed checkpoint broadcasts use pickle protocol 2 with restricted
loading on receivers; every rank must run the updated code.

The learning-rate schedulers now produce built-in floats. Checkpoint writers and
broadcast senders normalize producer-owned NumPy scalars into built-in values,
so ordinary optimizer/scheduler state can be resumed with restricted loading.
Existing files containing NumPy scalars still require trusted offline migration:
after independently verifying provenance, have the producer load the legacy file
in an isolated environment and export it with `save_weights` from
`cosmos_predict2._src.imaginaire.utils.checkpoint_loading`. Do not add an unsafe
fallback to the runtime reader.

A checkpoint requiring custom classes must be converted by its trusted producer
in an isolated environment. Export tensors/state dictionaries, or use the
existing safetensors model-loader path. Never disable restricted loading or
broadly allowlist classes to make an unknown checkpoint load. Restricted loading
is not a sandbox or proof of provenance; obtain artifacts from approved producers
and verify independently supplied digests/signatures before admitting them.

**Distributed checkpoints have a separate trust boundary.** PyTorch's DCP readers
unpickle `.metadata` before tensor restoration. They are only suitable for
independently verified, trusted training artifacts, not arbitrary supplied files.
`scripts/convert_distcp_to_pt.py` refuses conversion unless `--trust-checkpoint`
is explicitly supplied, before downloading, loading, or replacing output files.
This flag acknowledges the verified provenance; it does not verify it or sandbox
deserialization. Prefer an offline, disposable environment without credentials.
Restricted loading of the converted `.pt` cannot protect the original DCP input.

## Dataset and diagnostic artifacts

Runtime pickle and gzip-pickle data loading is disabled. This affects legacy
WebDataset embeddings, item datasets, and serialized diagnostic data. The Open-H
LeRobot JSON/Parquet/video dataset format is unchanged.

The replacement `.cdata` format is a versioned ZIP archive containing
`metadata.json` and numeric `.npy` arrays loaded with pickle disabled. The schema
supports string-keyed dictionaries, lists, tuples, scalar values, numeric NumPy
arrays, and ordinary CPU-restored PyTorch tensors (including bfloat16). It rejects
custom classes, object arrays, invalid shapes, unknown members, repeated array
references, and oversized or deeply nested data. Limits are 256 MiB total,
1 MiB metadata, 10,000 arrays, 100,000 nodes, and depth 64. These are limits per
artifact; deployments still need aggregate job and storage limits.

Use `safe_data.dump(value, binary_stream)` / `safe_data.load(binary_stream)` from
`cosmos_predict2._src.imaginaire.utils.safe_data`, or `easy_io.dump/load` with a
`.cdata` filename. The `.gz` handler now wraps a safe data archive, not pickle.
Changing a filename does not convert its contents.

Regenerate artifacts using the updated producer. If regeneration is impossible,
verify the exact legacy artifact against independently approved provenance and
perform its one-time conversion in an isolated, disposable environment without
credentials or network access. Legacy unpickling can execute code; that step is
not provided by or silently invoked in the runtime loaders. Review the result's
schema and use `safe_data.dump` to export it. Retain an inventory linking source
and converted artifact digests. Update item filenames to `.cdata` and repack
WebDataset shards with `.cdata` entries, preserving their sample stems.

## YAML configuration

YAML readers accept data-only YAML and reject Python object tags, aliases,
documents over 4 MiB, nesting beyond 64 levels, or more than 100,000 parser events.
Expand anchors/aliases and replace object tags with ordinary data before loading.
LazyConfig requires a string-keyed mapping. Python configuration and explicitly
instantiated `_target_` values remain trusted executable configuration; safe YAML
parsing does not make arbitrary configured Python targets safe.
YAML writers expand shared acyclic containers without aliases and enforce the
reader's size, depth, and event limits before writing output. Cycles and unsafe
dumper overrides are rejected.

## Gradio access

All shipped launchers require `COSMOS_GRADIO_USERNAME` and a unique password of at
least 16 characters, supplied with `COSMOS_GRADIO_PASSWORD_FILE` (recommended, mounted
secret) or `COSMOS_GRADIO_PASSWORD`. Do not put credentials in command lines, source
control, logs, or public discussions. Anonymous API, upload, file, and queue access
is blocked by Gradio authentication. Each application callback additionally checks
the authenticated operator before generation, uploads, listing, viewing, or log access.

This is a **single trusted operator** service: that operator may read all configured
uploads, results, and the application log and execute inference. It does not provide
tenant isolation; use separate instances and storage for mutually untrusted users.
Bind to loopback (the default) behind a TLS proxy or SSH tunnel. Direct non-loopback
listeners, including container port forwarding, require `GRADIO_SSL_CERTFILE` and
`GRADIO_SSL_KEYFILE` pointing to provisioned certificate/key files. Keep proxy access
logs free of credentials. Do not enable public Gradio share links.

Python clients use `gradio_client.Client(url, auth=(username, password))`. Browser
users log in through Gradio. Existing automated clients must provision credentials
from their secret store. Unauthenticated health checks should target a separate
proxy health endpoint.

## File viewer

Viewer selections are now paths relative to the configured uploads directory.
Symlinks (including symlinked subdirectories), absolute paths, parent traversal,
and non-regular files are rejected when opened. Media is copied from that open
file into a private, content-addressed temporary snapshot before Gradio serves it;
changing the original path cannot change the served snapshot. Text/JSON reads are
limited to 8 MiB and media to 500 MiB. Snapshots are cleaned up when the process
exits; provision temporary storage quotas for long-running instances.

Raw upload/output directories and the live log are no longer allowed HTTP download
paths. Generated media, viewer previews, and prepared log downloads are served as
private snapshots. The upload API still returns `path` for inference requests and
additionally returns `download_path` for authenticated `/gradio_api/file=...`
downloads. Do not construct a download URL from the mutable `path` field.

The application creates private per-process Gradio upload/example caches before
constructing the UI; shared `GRADIO_TEMP_DIR`/`GRADIO_EXAMPLES_CACHE` overrides are
replaced. Run mutually untrusted processes under separate OS users. Snapshot and
cache permissions protect against other users and writers to the source roots,
not hostile processes running under the application's own identity. Custom
mounts must call `configure_file_serving()` before constructing components and
pass `**file_serving_options()` when mounting. This supplies both the snapshot
allowlist and the URL-path guard, which rejects aliases from mutable directories
before Gradio resolves symlinks. Do not re-add source directories.

## Nightly container launchers

The nightly image installs a venv-local `torchrun` using
`/opt/cosmos-venv/bin/python`. Its worker processes see the same editable project
installation as the entrypoint. Keep that venv's bin directory first on PATH;
calling the system `/usr/local/bin/torchrun` directly bypasses the environment.
