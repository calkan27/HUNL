"""
I iterate over line-oriented JSON shard files containing CFV training examples and yield
parsed objects with minimal validation. I optionally verify the expected schema string,
skipping mismatches without failing the stream.

Key class: CFVShardDataset with iter that yields dict records. Expected fields: schema
(e.g., cfv.v1), input_vector, target_v1, target_v2; optional meta captured upstream.

Inputs: list of file paths, schema_version string, and verify flag. Outputs: dicts
deserialized from each valid line.

Dependencies: stdlib os/json. Invariants: I never raise on a malformed line; I print a
short reason and continue so long training jobs do not stop on bad shards. Performance:
I read sequentially and avoid buffering entire files in memory.
"""

import os
import json


class CFVShardDataset:
	def __init__(
		self,
		shard_paths,
		schema_version: str = "cfv.v1",
		verify_schema: bool = True,
	):
		self.paths = list(shard_paths)
		self.schema_version = str(schema_version)
		self.verify = bool(verify_schema)

	def __iter__(self):
		for p in self.paths:
			if isinstance(p, str):
				path = p
			else:
				path = str(p)

			if os.path.isfile(path):
				if os.access(path, os.R_OK):
					with open(path, "r") as f:
						for line in f:
							if isinstance(line, str):
								txt = line.strip()
							else:
								txt = ""

							if txt:
								if txt.startswith("{"):
									if txt.endswith("}"):
										obj = json.loads(txt)

										if self.verify:
											if str(obj.get("schema", "")) == self.schema_version:
												yield obj
											else:
												print(f"[SKIP] Schema mismatch in {path}.")
										else:
											yield obj
									else:
										print(f"[SKIP] Malformed JSON (no closing brace) in {path}.")
								else:
									print(f"[SKIP] Non-JSON line in {path}.")
							else:
								print(f"[SKIP] Empty line in {path}.")
				else:
					print(f"[SKIP] No read permission for {path}.")
			else:
				print(f"[SKIP] File not found: {path}.")

